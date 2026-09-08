"""
Compare three sparse-FFT engines -- sfft1 (Hassanieh et al., random
hashing), decimated (fixed decimation + binary phase, no randomness), and
rffast (Pawar & Ramchandran's R-FFAST: CRT-guided multi-stage aliasing +
onion-peeling) -- used as pluggable S3CA backends on a common DSSS-BPSK
cyclostationary test signal, the same one used in the S3CA paper's own
Section IV-A validation.

Usage
-----
    python3 bpsk_compare.py                  # default kappa=50
    python3 bpsk_compare.py --kappa 100 --snr 10 --Np 32

`sfft1` and `decimated` share one power-of-two N (any N works for them, but
`decimated` additionally needs a power-of-two decimation factor D | N, which
a power-of-two N always supplies). `rffast` needs N to factor into 3
pairwise-coprime pieces near kappa (see rffast.choose_stage_bins) -- a
power-of-two N cannot supply that, so it runs on a *different*, nearby N
(three near-consecutive integers around kappa, e.g. 99*100*101 for
kappa=100 -- matching the R-FFAST paper's own example). Both N are printed
so the comparison's scale is transparent; see `three_coprime_near` for how
the rffast N is chosen and how close it ends up to the power-of-two one.

This module exposes `run_comparison(...)` returning a plain dict of
results (so the accompanying notebook can call it directly and build its
own plots/markdown around the numbers), plus a `__main__` block that prints
a report when run as a script.
"""

from __future__ import annotations

import argparse
from math import gcd
from dataclasses import dataclass, field

import numpy as np

from s3ca import check, print_report, dense_ssca, CheckReport
from demo_s3ca import dsss_bpsk


# ---------------------------------------------------------------------------
# Picking a compatible N for each backend
# ---------------------------------------------------------------------------
def three_coprime_near(kappa, max_span=12):
    """Three pairwise-coprime integers as close to `kappa` as possible
    (searching outward symmetrically), for rffast's stage-bin counts.
    Consecutive integers are always pairwise coprime except for shared even
    factors, so `[kappa-1, kappa, kappa+1]` works whenever kappa is odd;
    for even kappa this searches a little wider (matching what
    `rffast.choose_stage_bins` would find on n = product of the three)."""
    best = None
    for span in range(0, max_span):
        cand = list(range(max(2, kappa - span), kappa + span + 1))
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                for k in range(j + 1, len(cand)):
                    a, b, c = cand[i], cand[j], cand[k]
                    if gcd(a, b) == gcd(a, c) == gcd(b, c) == 1:
                        cost = abs(a - kappa) + abs(b - kappa) + abs(c - kappa)
                        if best is None or cost < best[0]:
                            best = (cost, sorted([a, b, c]))
        if best is not None:
            return best[1]
    raise RuntimeError(f"no coprime triple found near kappa={kappa}")


def nearest_pow2(n):
    return 1 << round(np.log2(n))


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def bpsk_signal(n_samples, chip_rate=0.25, gain=31, snr_db=10.0, seed=0):
    """DSSS-BPSK, truncated/padded to exactly `n_samples`. Matches the
    S3CA paper's own Sec. IV-A test signal (chip_rate=0.25, gain=31,
    snr_db=10) -- see s3ca_spl24.pdf and demo_s3ca.dsss_bpsk.

    Returns (x, data_rate) where data_rate = chip_rate/gain is the
    fundamental cycle frequency (normalised to fs=1), used to grade
    recovered peaks against known physics via `check()`'s
    `expected_data_rate`.
    """
    sps = round(1.0 / chip_rate)
    n_symbols = n_samples // (sps * gain) + 1
    x, data_rate = dsss_bpsk(n_symbols, chip_rate=chip_rate, gain=gain,
                              snr_db=snr_db, rng=seed)
    return x[:n_samples], data_rate


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
@dataclass
class ComparisonResult:
    kappa: int
    Np: int
    snr_db: float
    data_rate: float
    N_pow2: int
    N_rffast: int
    rffast_f: list
    reports: dict = field(default_factory=dict)   # backend name -> CheckReport
    signals: dict = field(default_factory=dict)   # backend name -> x used
    dense: dict = field(default_factory=dict)     # backend name -> (S,f,alpha) dense SSCA


def run_comparison(kappa=50, Np=32, snr_db=10.0, chip_rate=0.25, gain=31,
                    seed=0, alpha_tol_bins=2,
                    sfft1_kwargs=None, decimated_kwargs=None,
                    rffast_kwargs=None) -> ComparisonResult:
    """Run sfft1, decimated, and rffast (each as an S3CA backend, mode
    "full") on the same DSSS-BPSK scenario, and return everything needed
    to report or plot the comparison.

    Each backend uses its own N (see module docstring); `kappa`, `Np`,
    `snr_db`, `chip_rate`, `gain`, `seed` are otherwise held fixed across
    the three so the comparison isolates the sparse-FFT engine, not the
    scenario.
    """
    sfft1_kwargs = sfft1_kwargs or dict(loc_loops=4, est_loops=8, tolerance=1e-4)
    decimated_kwargs = decimated_kwargs or dict(D=16, window="kaiser", beta=12.0)

    rffast_f = three_coprime_near(kappa)
    N_rffast = rffast_f[0] * rffast_f[1] * rffast_f[2]
    N_pow2 = nearest_pow2(N_rffast)

    rffast_kwargs = dict(rffast_kwargs or {})
    rffast_kwargs.setdefault("f", rffast_f)
    rffast_kwargs.setdefault("per_cluster", 3)

    result = ComparisonResult(kappa=kappa, Np=Np, snr_db=snr_db,
                               data_rate=chip_rate / gain, N_pow2=N_pow2,
                               N_rffast=N_rffast, rffast_f=rffast_f)

    x_pow2, dr = bpsk_signal(N_pow2, chip_rate, gain, snr_db, seed)
    x_rffast, dr2 = bpsk_signal(N_rffast, chip_rate, gain, snr_db, seed)
    assert abs(dr - dr2) < 1e-15
    result.data_rate = dr

    backend_signal = {"sfft1": x_pow2, "decimated": x_pow2, "rffast": x_rffast}
    backend_kwargs = {"sfft1": sfft1_kwargs, "decimated": decimated_kwargs,
                       "rffast": rffast_kwargs}

    for name, x in backend_signal.items():
        r = check(x, Np, kappa, backend=name, seed=seed,
                  expected_data_rate=dr, alpha_tol_bins=alpha_tol_bins,
                  **backend_kwargs[name])
        result.reports[name] = r
        result.signals[name] = x

    return result


def run_amtone_comparison(kappa=40, Np=32, snr_db=30.0, f0=0.05,
                           cycle_bin=37.0, seed=1,
                           sfft1_kwargs=None, decimated_kwargs=None,
                           rffast_kwargs=None) -> ComparisonResult:
    """Companion scenario: an amplitude-modulated tone (demo_s3ca.am_tone),
    genuinely sparse in cycle frequency per channel -- unlike DSSS-BPSK,
    this is decimated_sfft's/rffast's home turf (a clean singleton-test
    regime), included so the comparison isn't one-sided. See the notebook
    for why the two scenarios pull the three backends apart differently.
    """
    from demo_s3ca import am_tone

    sfft1_kwargs = sfft1_kwargs or dict(loc_loops=4, est_loops=8, tolerance=1e-4)
    decimated_kwargs = decimated_kwargs or dict(D=16, window="kaiser", beta=12.0)

    rffast_f = three_coprime_near(kappa)
    N_rffast = rffast_f[0] * rffast_f[1] * rffast_f[2]
    N_pow2 = nearest_pow2(N_rffast)

    rffast_kwargs = dict(rffast_kwargs or {})
    rffast_kwargs.setdefault("f", rffast_f)
    rffast_kwargs.setdefault("per_cluster", 3)

    result = ComparisonResult(kappa=kappa, Np=Np, snr_db=snr_db,
                               data_rate=cycle_bin / N_pow2, N_pow2=N_pow2,
                               N_rffast=N_rffast, rffast_f=rffast_f)

    alpha0_pow2 = cycle_bin / N_pow2
    alpha0_rffast = cycle_bin / N_rffast
    x_pow2 = am_tone(N_pow2, alpha0_pow2, f0=f0, snr_db=snr_db, rng=seed)
    x_rffast = am_tone(N_rffast, alpha0_rffast, f0=f0, snr_db=snr_db, rng=seed)

    backend_signal = {"sfft1": (x_pow2, alpha0_pow2),
                       "decimated": (x_pow2, alpha0_pow2),
                       "rffast": (x_rffast, alpha0_rffast)}
    backend_kwargs = {"sfft1": sfft1_kwargs, "decimated": decimated_kwargs,
                       "rffast": rffast_kwargs}

    for name, (x, alpha0) in backend_signal.items():
        r = check(x, Np, kappa, backend=name, seed=seed,
                  expected_data_rate=alpha0, alpha_tol_bins=2,
                  **backend_kwargs[name])
        result.reports[name] = r
        result.signals[name] = x

    return result


def comparison_table(result: ComparisonResult):
    """Return (backends, metric_name -> list of values) for plotting."""
    names = list(result.reports)
    metrics = {
        "hit rate": [result.reports[n].full_hit_rate for n in names],
        "fill rate": [result.reports[n].fill_rate for n in names],
        "alpha grid": [(result.reports[n].alpha_grid_hit_rate or 0.0) for n in names],
        "speedup (x, right axis)": [result.reports[n].full_speedup for n in names],
    }
    return names, metrics


def print_comparison_report(result: ComparisonResult) -> None:
    print(f"kappa={result.kappa}, Np={result.Np}, SNR={result.snr_db} dB, "
          f"data_rate={result.data_rate:.6g}")
    print(f"  sfft1/decimated N   = {result.N_pow2} (2^{np.log2(result.N_pow2):.0f})")
    print(f"  rffast N            = {result.N_rffast} "
          f"(f = {result.rffast_f}, {100*(result.N_rffast/result.N_pow2-1):+.1f}% "
          f"vs the power-of-two N)")
    print()
    header = (f"{'backend':12}{'N':>10}{'dense(ms)':>11}{'full(ms)':>10}"
              f"{'speedup':>9}{'hit rate':>10}{'fill rate':>11}{'alpha grid':>12}")
    print(header)
    for name, r in result.reports.items():
        N = result.N_pow2 if name != "rffast" else result.N_rffast
        agr = f"{r.alpha_grid_hit_rate:.1%}" if r.alpha_grid_hit_rate is not None else "--"
        print(f"{name:12}{N:>10}{r.dense_time*1e3:>11.1f}{r.full_time*1e3:>10.1f}"
              f"{r.full_speedup:>8.2f}x{r.full_hit_rate:>10.1%}"
              f"{r.fill_rate:>11.1%}{agr:>12}")
        if r.full_extra:
            print(f"  {name} diagnostics: {r.full_extra}")
    print()
    print("(hit rate: fraction of the dense top-kappa peaks each backend recovered.")
    print(" fill rate: recovered coefficients / (kappa*Np) -- low fill rate for a")
    print(" singleton-style decoder (decimated, rffast) usually means it is correctly")
    print(" abstaining on multi-ton bins, not guessing wrong -- see the notebook for")
    print(" why DSSS-BPSK's CDP is a genuinely hard case for that style of decoder.")
    print(" alpha grid: fraction of recovered cycle frequencies landing near an")
    print(" integer multiple of the known data rate -- a check against independently")
    print(" known physics, not just against the dense baseline.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kappa", type=int, default=50)
    ap.add_argument("--Np", type=int, default=32)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--chip-rate", type=float, default=0.25)
    ap.add_argument("--gain", type=int, default=31)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    result = run_comparison(kappa=args.kappa, Np=args.Np, snr_db=args.snr,
                             chip_rate=args.chip_rate, gain=args.gain,
                             seed=args.seed)
    print_comparison_report(result)
