"""
fam.py -- a FAM-style (FFT Accumulation Method) cyclic spectrum estimator,
built by reusing s3ca.py's channelizer/CDP machinery and pluggable
sparse-FFT backend protocol, but evaluating the channelizer only at every
L-th raw sample instead of every sample.

------------------------------------------------------------------------------
Honest scope note -- read this before trusting the numbers
------------------------------------------------------------------------------
This captures FAM's one property this whole investigation cares about: the
cycle-frequency-resolving FFT runs on a SHORT "slow time" sequence of length
P = N//L, not on the full record length N the way dense_ssca/s3ca's outer
FFT does. That's a real, structural difference, not a parameter tweak --
it's exactly the "smaller FFT" property that makes an aliasing-based
sparse-FFT engine like rffast a structurally better fit here than for
SSCA's big N-point FFT (see the aliasing-loss discussion in the S3CA
notebook this project already built: loss scales with P_s*n, and shrinking
the transform length n shrinks it roughly quadratically for fixed sparsity
kappa).

This is *not* a bit-for-bit reproduction of the classical Roberts, Brown &
Loomis (1991) FAM algorithm, which uses explicit bin-pair (k1,k2) spectral
products, specific block-window/overlap conventions, and a non-uniform
frequency-plane mapping this module does not attempt to replicate. What's
reused instead is the exact same channelizer/CDP construction as
s3ca.dense_ssca -- X_T(t,k)*conj(x(t))*g(t) -- just evaluated at strided
positions t = 0, L, 2L, ... instead of every t. This is a legitimate,
simplified way to get FAM's essential computational shape (channelizer
feeding a much shorter slow-time FFT) without re-deriving FAM's own
bin-pairing formalism, but the resemblance to "real" FAM is architectural,
not exact.

The cost of the simplification, stated plainly: resolvable cycle frequency
is limited to |alpha| < 1/(2L) (the Nyquist rate of the decimated slow-time
axis), versus the full range dense_ssca's N-point outer FFT covers. This is
a real, disclosed trade-off -- not a free lunch -- and matches the
literature's own characterization of FAM's reduced/non-uniform resolution
relative to SSCA (see S3CA paper's own related-work discussion).
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

from s3ca import (_centered, _default_windows, _channelizer_rows,
                   _channelizer_footprint, _to_f_alpha, make_backend)

__all__ = ["dense_fam", "fam", "FAMResult", "check_fam", "FAMCheckReport",
           "print_fam_report", "compare_fam_backends", "print_fam_comparison"]


# ---------------------------------------------------------------------------
# dense FAM -- ground truth
# ---------------------------------------------------------------------------
def dense_fam(x, Np, L, a=None, g=None):
    """Dense (ground-truth) FAM-style estimate. Channelizer evaluated every
    L raw samples (P = N//L slow-time positions per channel), then a dense
    P-point FFT resolves cycle frequency.

    Returns
    -------
    S : complex array, shape (P, Np).
    f, alpha : real arrays, shape (P, Np).
    P : the slow-time axis length actually used.
    """
    x = np.asarray(x)
    N = x.size
    P = N // L
    if P < 2:
        raise ValueError(f"L={L} too large for N={N}: P=N//L={P} < 2")
    t_idx = (np.arange(P) * L).astype(np.int64)

    if a is None:
        a, _ = _default_windows(N, Np)
    if g is None:
        g = np.hamming(P)

    XT, k_idx = _channelizer_rows(x, Np, a, t_idx)          # (P, Np)
    Xg = XT * np.conj(x[t_idx])[:, None] * g[:, None]

    Sfft = np.fft.fft(Xg, axis=0)
    q_idx = _centered(P)
    S = Sfft[q_idx % P, :]

    f, alpha = _to_f_alpha(k_idx[None, :], q_idx[:, None], Np, P * L)
    return S, f, alpha, P


# ---------------------------------------------------------------------------
# sparse FAM
# ---------------------------------------------------------------------------
class FAMResult(NamedTuple):
    f: np.ndarray
    alpha: np.ndarray
    value: np.ndarray
    channel: np.ndarray
    P: int                    # slow-time axis length (the backend's own "N")
    n_positions: int          # how many slow-time positions the channelizer/CDP
                               # were evaluated at -- P in "naive" mode, |W'| << P
                               # in "full" mode (this is FAM's analogue of
                               # S3CAResult.n_positions, just on the P-axis)
    n_raw_samples_read: int   # distinct raw x[] samples read to do that
    elapsed: float
    backend: str
    extra: dict


def fam(x, Np, L, kappa, mode="full", seed=0, a=None, g=None,
        backend="sfft1", **backend_kwargs) -> FAMResult:
    """Sparse FAM: same shape as s3ca.s3ca(), but the pluggable backend
    operates on the P = N//L -length *slow-time* axis instead of the full
    record -- see module docstring for why that's the point.
    """
    x = np.asarray(x)
    N = x.size
    P = N // L
    if P < 2:
        raise ValueError(f"L={L} too large for N={N}: P=N//L={P} < 2")
    t_idx_all = (np.arange(P) * L).astype(np.int64)

    if a is None:
        a, _ = _default_windows(N, Np)
    if g is None:
        g = np.hamming(P)

    be = backend if not isinstance(backend, str) else make_backend(
        P, kappa, backend=backend, **backend_kwargs)

    k_idx = _centered(Np)
    t0 = time.perf_counter()

    fs, alphas, values, chans = [], [], [], []
    extra_total: dict = {}

    def _accumulate(extra):
        for key, val in extra.items():
            extra_total[key] = extra_total.get(key, 0) + val

    if mode == "naive":
        XT, _ = _channelizer_rows(x, Np, a, t_idx_all)
        Xg = XT * np.conj(x[t_idx_all])[:, None] * g[:, None]
        n_positions = P
        n_raw = _channelizer_footprint(t_idx_all, N, Np).size
        for ci, k in enumerate(k_idx):
            freqs, coeffs, extra = be.run(Xg[:, ci], kappa,
                                           seed=None if seed is None else seed + ci)
            _accumulate(extra)
            q_centered = np.where(freqs < P // 2, freqs, freqs - P)
            f, alpha = _to_f_alpha(k, q_centered, Np, P * L)
            fs.append(f); alphas.append(alpha); values.append(coeffs)
            chans.append(np.full(freqs.size, k))

    elif mode == "full":
        req = be.required_indices(seed)              # indices into [0, P)
        n_positions = req.size
        t_idx = t_idx_all[req]                          # map back to raw sample positions
        n_raw = _channelizer_footprint(t_idx, N, Np).size

        XT_rows, _ = _channelizer_rows(x, Np, a, t_idx)
        Xg_rows = XT_rows * np.conj(x[t_idx])[:, None] * g[req][:, None]

        for ci, k in enumerate(k_idx):
            Xg_col = np.zeros(P, dtype=complex)
            Xg_col[req] = Xg_rows[:, ci]
            freqs, coeffs, extra = be.run(Xg_col, kappa, seed=seed)
            _accumulate(extra)
            q_centered = np.where(freqs < P // 2, freqs, freqs - P)
            f, alpha = _to_f_alpha(k, q_centered, Np, P * L)
            fs.append(f); alphas.append(alpha); values.append(coeffs)
            chans.append(np.full(freqs.size, k))
    else:
        raise ValueError(f"mode must be 'naive' or 'full', got {mode!r}")

    elapsed = time.perf_counter() - t0
    f = np.concatenate(fs) if fs else np.empty(0)
    alpha = np.concatenate(alphas) if alphas else np.empty(0)
    value = np.concatenate(values) if values else np.empty(0, dtype=complex)
    channel = np.concatenate(chans) if chans else np.empty(0, dtype=int)
    return FAMResult(f, alpha, value, channel, P, n_positions, n_raw, elapsed,
                      be.name, extra_total)


# ---------------------------------------------------------------------------
# checking
# ---------------------------------------------------------------------------
class FAMCheckReport(NamedTuple):
    backend: str
    kappa: int
    Np: int
    N: int            # raw record length
    L: int             # slow-time stride
    P: int             # slow-time axis length = N // L
    dense_time: float
    naive_time: float
    full_time: float
    naive_hit_rate: float
    full_hit_rate: float
    naive_residual: float
    full_residual: float
    full_sparsity_ratio: float          # |W'| / P (FAM's analogue of S3CA's N-based one)
    full_raw_sample_ratio: float        # distinct raw x[] samples read / N
    naive_style_sparsity_ratio: float
    naive_speedup: float
    full_speedup: float
    alpha_grid_hit_rate: float | None
    fill_rate: float
    full_extra: dict


def _support_and_residual_fam(S_dense, k_idx, q_idx, result: FAMResult,
                               kappa, Np, P, L):
    """s3ca._support_and_residual, adapted to FAM's P-length slow-time axis
    and its P*L-scaled cycle-frequency resolution (dalpha = 1/(P*L), not
    1/P) -- see module docstring."""
    mag = np.abs(S_dense)
    k_pos = {int(k): i for i, k in enumerate(k_idx)}
    dalpha = 1.0 / (P * L)

    true_sets = {}
    n_true = 0
    for ki in range(Np):
        kk = min(kappa, mag.shape[0])
        top = np.argpartition(-mag[:, ki], kk - 1)[:kk]
        true_sets[ki] = set(top.tolist())
        n_true += kk

    recon = np.zeros_like(S_dense)
    hit = 0
    for kk, al, val in zip(result.channel, result.alpha, result.value):
        ki = k_pos[int(kk)]
        fk = k_idx[ki] / Np
        q = round((al - fk) / dalpha)
        qi = int(np.searchsorted(q_idx, q))
        if qi >= len(q_idx) or q_idx[qi] != q:
            continue
        recon[qi, ki] = val
        if qi in true_sets[ki]:
            hit += 1
    hit_rate = hit / n_true if n_true else float("nan")
    residual = np.sum(np.abs(S_dense - recon)) / (kappa * Np)
    return hit_rate, residual


def check_fam(x, Np, L, kappa, a=None, g=None, backend="sfft1", seed=0,
              expected_data_rate=None, alpha_tol_bins=1,
              **backend_kwargs) -> FAMCheckReport:
    """FAM's analogue of s3ca.check(): run dense_fam and both sparse-FAM
    modes on the same `x`, report hit rate / residual / timing / sparsity
    against the dense FAM-style reference (not against dense_ssca -- FAM
    and SSCA resolve different, generally non-identical, cycle-frequency
    grids, so comparing sparse-FAM to dense-FAM is the fair, apples-to-
    apples check here).
    """
    x = np.asarray(x)
    N = x.size
    P = N // L
    if a is None:
        a, _ = _default_windows(N, Np)
    if g is None:
        g = np.hamming(P)

    t0 = time.perf_counter()
    S_dense, _, _, P = dense_fam(x, Np, L, a=a, g=g)
    dense_time = time.perf_counter() - t0

    be = backend if not isinstance(backend, str) else make_backend(
        P, kappa, backend=backend, **backend_kwargs)

    k_idx = _centered(Np)
    q_idx = _centered(P)

    naive = fam(x, Np, L, kappa, mode="naive", seed=seed, a=a, g=g, backend=be)
    full = fam(x, Np, L, kappa, mode="full", seed=seed, a=a, g=g, backend=be)

    naive_hit, naive_res = _support_and_residual_fam(S_dense, k_idx, q_idx, naive, kappa, Np, P, L)
    full_hit, full_res = _support_and_residual_fam(S_dense, k_idx, q_idx, full, kappa, Np, P, L)

    union = set()
    for ci in range(Np):
        idx_ci = be.required_indices(None if seed is None else seed + ci)
        union.update(idx_ci.tolist())
    naive_style_ratio = len(union) / P

    alpha_grid_hit = None
    if expected_data_rate is not None and full.alpha.size:
        m = np.round(full.alpha / expected_data_rate)
        nearest = m * expected_data_rate
        tol = alpha_tol_bins / (P * L)
        alpha_grid_hit = float(np.mean(np.abs(full.alpha - nearest) <= tol))

    return FAMCheckReport(
        backend=be.name, kappa=kappa, Np=Np, N=N, L=L, P=P,
        dense_time=dense_time, naive_time=naive.elapsed, full_time=full.elapsed,
        naive_hit_rate=naive_hit, full_hit_rate=full_hit,
        naive_residual=naive_res, full_residual=full_res,
        full_sparsity_ratio=full.n_positions / P,
        full_raw_sample_ratio=full.n_raw_samples_read / N,
        naive_style_sparsity_ratio=naive_style_ratio,
        naive_speedup=dense_time / naive.elapsed if naive.elapsed else float("inf"),
        full_speedup=dense_time / full.elapsed if full.elapsed else float("inf"),
        alpha_grid_hit_rate=alpha_grid_hit,
        fill_rate=full.channel.size / (kappa * Np),
        full_extra=full.extra,
    )


def compare_fam_backends(x, Np, L, kappa, backends=("sfft1", "decimated", "rffast"),
                          backend_kwargs=None, **common_kw) -> dict:
    backend_kwargs = backend_kwargs or {}
    return {b: check_fam(x, Np, L, kappa, backend=b, **common_kw, **backend_kwargs.get(b, {}))
            for b in backends}


def print_fam_report(r: FAMCheckReport) -> None:
    print(f"backend = {r.backend!r}, N = {r.N} raw samples, L = {r.L} (slow-time stride), "
          f"P = N/L = {r.P}, Np = {r.Np}, kappa = {r.kappa}")
    print(f"max resolvable |alpha| = 1/(2L) = {1/(2*r.L):.4g} "
          f"(vs. dense_ssca's full [-1,1] range -- this is FAM's real trade-off)")
    print(f"{'':16}{'time (ms)':>12}{'speedup':>10}{'hit rate':>11}{'residual':>11}")
    print(f"{'dense FAM':16}{r.dense_time*1e3:12.1f}{'--':>10}{'--':>11}{'--':>11}")
    print(f"{'naive FAM':16}{r.naive_time*1e3:12.1f}{r.naive_speedup:9.2f}x"
          f"{r.naive_hit_rate:11.1%}{r.naive_residual:11.3g}")
    print(f"{'FAM (full)':16}{r.full_time*1e3:12.1f}{r.full_speedup:9.2f}x"
          f"{r.full_hit_rate:11.1%}{r.full_residual:11.3g}")
    print(f"FAM (full) evaluated the channelizer/CDP at {r.full_sparsity_ratio:.1%} of "
          f"the P={r.P} slow-time positions")
    print(f"  -- vs. {r.naive_style_sparsity_ratio:.1%} for {r.Np} independently-deciding "
          f"channels ({'no COMPIDX gap' if abs(r.naive_style_sparsity_ratio - r.full_sparsity_ratio) < 1e-9 else 'naive fragments'})")
    if r.alpha_grid_hit_rate is not None:
        print(f"FAM (full) recovered peaks landing on the expected cycle-frequency grid: "
              f"{r.alpha_grid_hit_rate:.1%}")
    if r.full_extra:
        print(f"FAM (full) backend diagnostics: {r.full_extra}")


def print_fam_comparison(reports: dict) -> None:
    names = list(reports)
    r0 = reports[names[0]]
    namew = max(12, max(len(n) for n in names) + 1)
    print(f"N={r0.N} raw samples, L={r0.L}, P=N/L={r0.P}, Np={r0.Np}, kappa={r0.kappa} "
          f"(dense FAM reference: {r0.dense_time*1e3:.1f} ms)")
    header = (f"{'backend':{namew}}{'hit rate':>10}{'residual':>11}{'speedup':>9}"
              f"{'|Wprime|/P':>11}{'fill rate':>11}{'alpha grid':>12}")
    print(header)
    for name in names:
        r = reports[name]
        agr = f"{r.alpha_grid_hit_rate:.1%}" if r.alpha_grid_hit_rate is not None else "--"
        print(f"{name:{namew}}{r.full_hit_rate:10.1%}{r.full_residual:11.3g}"
              f"{r.full_speedup:8.2f}x{r.full_sparsity_ratio:11.1%}{r.fill_rate:11.1%}{agr:>12}")
        if r.full_extra:
            print(f"{'':{namew}}{r.full_extra}")
