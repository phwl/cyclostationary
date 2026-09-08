"""
harmonic.py -- harmonic-structured recovery for the strip spectral
correlation analyzer.

------------------------------------------------------------------------------
The idea
------------------------------------------------------------------------------
Every sparse-FFT backend tried so far (sfft1, decimated, rffast) treats
recovering a channel's cyclic spectrum as "find kappa arbitrary-location
nonzero coefficients out of N candidates" -- a generic sparse-recovery
problem. But a cyclostationary signal's significant cycle frequencies are
*not* arbitrary: for a linearly modulated digital signal they sit at
alpha = 0 (ordinary power) and alpha = +-m*alpha0 for small integers m,
where alpha0 is the symbol/chip rate (Gardner's cyclostationarity theory).
That's a MUCH lower-dimensional problem: find one scalar (alpha0), then
directly evaluate amplitudes at the now-known harmonic locations.

Three-stage pipeline, each stage validated independently before trusting
the whole thing (see the exploratory session that built this: a blind
periodogram search for alpha0 was tried first and found to be genuinely
fragile -- with few sparse samples, random sidelobes can exceed the true
peak's strength, a classic problem with magnitude/energy-based frequency
search from an under-resolved aperture):

1. **Candidate pool.** Run an existing, already-validated backend (sfft1
   by default) at an *elevated* kappa (kappa_search > kappa) to get a
   richer-than-needed candidate list per channel, pooled across all Np
   channels (alpha0 is a global property of the signal, common to every
   channel, so pooling gives much more evidence than any one channel
   alone).

2. **Fundamental search.** sfft1 recovers exact DFT bin indices, so every
   candidate's cycle frequency is alpha_i = q_i/N for integer q_i. If a
   true fundatmental q0 exists, every genuine harmonic's q_i is close to
   an integer multiple of q0 -- so q0 is found by scoring candidate q0
   values (weighted by |value|) on how many pooled candidates they
   explain as a small-integer multiple, THE SAME STRUCTURE AS FINDING A
   COMMON DIVISOR, just tolerant of noise and spurious candidates. This
   needs a caller-supplied plausible q0 range (`q0_range`): searching an
   unconstrained range is ill-posed (q0=1 trivially "explains" every
   integer) in exactly the way looking for *some* period without any
   prior always is -- a real system has *some* idea of the plausible baud
   rate range, and this makes that assumption explicit rather than
   hiding a fragile auto-detected one.

3. **Joint harmonic refit.** Given q0, build the candidate harmonic comb
   {0, +-q0, +-2*q0, ..., +-n_harmonics*q0} and, per channel, solve a
   small joint least-squares fit for the amplitude at every comb
   frequency simultaneously, using the *same* samples already read for
   stage 1 (no new reads). This is what actually fixes two of the
   concrete failure modes diagnosed earlier in this project:
   cross-talk between nearby recovered frequencies (independent
   per-candidate estimation, as sfft1/rffast both do, doesn't deconflict
   them; a joint fit does), and coefficients sfft1's blind search missed
   entirely (once q0 is known, every harmonic is *directly evaluated*,
   not searched for, so a weak-but-real harmonic that lost sfft1's
   internal voting can still be recovered here).

------------------------------------------------------------------------------
Honest limitations
------------------------------------------------------------------------------
* Single fundamental only. Multiple overlapping cyclostationary signals
  (each with its own baud rate) would need a peel-and-repeat extension
  (find the strongest q0, subtract its fitted comb's contribution from
  the candidate pool, repeat) -- not implemented here.
* q0_range must be supplied. This is a stated design choice (see stage 2
  above), not an oversight -- pass the plausible baud-rate range you'd
  actually have for signals of interest.
* Octave/sub-multiple ambiguity is possible (q0 vs 2*q0 or q0/2) if the
  true comb happens to be sparse in a way a multiple also explains well.
  `estimate_fundamental` returns the top few scoring candidates so this
  can be inspected; it does not attempt automatic disambiguation beyond
  the score margin.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

from s3ca import (_centered, _default_windows, _channelizer_rows,
                   _channelizer_footprint, _to_f_alpha, make_backend,
                   S3CAResult)

__all__ = ["estimate_fundamental", "fit_harmonics", "harmonic_ssca",
           "check_harmonic", "HarmonicCheckReport", "print_harmonic_report"]


# ---------------------------------------------------------------------------
# Stage 2: fundamental search from a pooled candidate list
# ---------------------------------------------------------------------------
def estimate_fundamental(qs, weights, q0_range, top_k=5, tie_margin=0.05):
    """Find the integer q0 (cycle-frequency bin spacing) best supported by
    a pooled candidate list (qs, weights) -- see module docstring, stage 2.

    Parameters
    ----------
    qs : integer array, candidate bin indices (exclude q=0/DC before
        calling -- DC is not informative about the fundamental spacing).
    weights : array, same length, e.g. |value| or |value|^2 per candidate
        -- stronger candidates count more.
    q0_range : range or array of candidate q0 values to score. Required;
        see module docstring for why this can't be auto-inferred safely.
    top_k : how many top-scoring candidates to return, for inspecting
        octave/sub-multiple ambiguity.

    Returns
    -------
    best_q0 : int
    ranked : list of (q0, score) for the top_k candidates, best first.
    """
    qs = np.asarray(qs)
    weights = np.asarray(weights, dtype=float)
    scores = {}
    for q0 in q0_range:
        if q0 <= 0:
            continue
        m = np.round(qs / q0)
        m[m == 0] = 1          # a candidate can't be its own 0th harmonic
        resid = np.abs(qs - m * q0)
        scores[int(q0)] = float(np.sum(weights[resid == 0]))
    if not scores:
        raise ValueError("q0_range produced no candidates to score")
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]

    # Subharmonic disambiguation: a candidate q0_small that is an exact
    # divisor of a comparably-scoring q0_large is often an artifact (the
    # small candidate trivially "explains" every harmonic of the large
    # one too, at multiples of the divisor) rather than the true
    # fundamental -- found directly on DSSS-BPSK, where q0 and q0/7 tied
    # within a score margin of 1%. Prefer the largest q0 among near-tied
    # top candidates that are exact multiples of each other.
    best_q0, best_score = ranked[0]
    for q0, score in ranked[1:]:
        if score < best_score * (1.0 - tie_margin):
            break
        if q0 > best_q0 and q0 % best_q0 == 0:
            best_q0 = q0
    return best_q0, ranked


# ---------------------------------------------------------------------------
# Stage 1 + 3 driver
# ---------------------------------------------------------------------------
class HarmonicResult(NamedTuple):
    f: np.ndarray
    alpha: np.ndarray
    value: np.ndarray
    channel: np.ndarray
    n_positions: int
    n_raw_samples_read: int
    elapsed: float
    backend: str
    extra: dict           # {"alpha0": ..., "q0": ..., "ranked_q0": [...]}


def fit_harmonics(Xg_sparse, t_idx, N, q0, n_harmonics, k_idx, Np):
    """Stage 3: joint per-channel least-squares fit at global cycle
    frequencies {0, +-q0, ..., +-n_harmonics*q0} (in units of 1/N), using
    the samples already read (Xg_sparse, shape (len(t_idx), Np)).

    Two things worth being explicit about, both found the hard way while
    building this (see module history):

    1. Sign convention. Sfft = fft(Xg) uses numpy's convention (negative
       exponent), so the *inverse* relationship this fit is built on --
       expressing Xg(t) as a sum of its Fourier components -- needs the
       opposite (positive) sign: Xg(t) ~ sum_q Sfft[q] * exp(+i*2*pi*t*q/N).

    2. Channel offset. Xg(t,k)'s own down-conversion (the channelizer
       applies exp(-i*2*pi*f_k*t) internally) means the outer-FFT bin
       matching global cycle frequency alpha for channel k is
       q_rel = round((alpha - f_k)*N), not round(alpha*N) -- i.e. every
       channel needs its OWN basis, shifted by -f_k*N = -k*N/Np. Skipping
       this (using the same global-alpha frequency for every channel)
       silently fits the wrong tone per channel and gives badly-ranked,
       wrong-by-orders-of-magnitude amplitudes -- exactly what the first
       version of this function produced.
    """
    m_list = np.array(range(-n_harmonics, n_harmonics + 1))
    global_q = m_list * q0                      # global alpha-space index, shared meaning
    Np_ = Np
    values = np.empty((len(m_list), Np_), dtype=complex)
    for k in range(Np_):
        fk_N = k_idx[k] * N / Np_               # = f_k * N; exact for N % Np == 0
        q_rel = global_q - fk_N
        A = np.exp(1j * 2 * np.pi * np.outer(t_idx, q_rel) / N)
        v, *_ = np.linalg.lstsq(A, Xg_sparse[:, k], rcond=None)
        values[:, k] = v
    # The model is Xg(t) ~ sum_q Sfft[q]*exp(+i*2*pi*t*q/N)/N (the actual
    # IDFT, 1/N and all), so the least-squares fit recovers Sfft[q]/N, not
    # Sfft[q] itself -- rescale to match dense_ssca's own convention
    # (raw fft(Xg), no 1/N), confirmed against it directly: the fit's
    # ranking was already correct before this fix, only the overall scale
    # was off by exactly N.
    values *= N
    return global_q, values


def harmonic_ssca(x, Np, kappa, q0_range, n_harmonics=None, kappa_search_mult=3,
                   base_backend="sfft1", base_backend_kwargs=None,
                   seed=0, a=None, g=None) -> HarmonicResult:
    """Harmonic-structured S3CA recovery -- see module docstring.

    kappa_search_mult : the candidate-pool stage searches for
        kappa*kappa_search_mult per channel (a richer pool than the final
        target), since the fundamental search benefits from more
        evidence and the final output is capped to kappa anyway.
    n_harmonics : comb half-width; defaults to enough to reach kappa
        candidates (including DC): n_harmonics = max(1, (kappa-1)//2).
    """
    x = np.asarray(x)
    N = x.size
    if a is None or g is None:
        a_def, g_def = _default_windows(N, Np)
        a = a_def if a is None else a
        g = g_def if g is None else g
    if n_harmonics is None:
        n_harmonics = max(1, (kappa - 1) // 2)
    base_backend_kwargs = base_backend_kwargs or {}

    t0 = time.perf_counter()
    k_idx = _centered(Np)

    # Stage 1: candidate pool via an existing backend, elevated kappa.
    kappa_search = max(kappa * kappa_search_mult, 2 * n_harmonics + 1)
    be = make_backend(N, kappa_search, backend=base_backend, **base_backend_kwargs)
    Wp_idx = be.required_indices(seed)
    n_positions = Wp_idx.size
    n_raw = _channelizer_footprint(Wp_idx, N, Np).size

    XT_rows, _ = _channelizer_rows(x, Np, a, Wp_idx)
    Xg_rows = XT_rows * np.conj(x[Wp_idx])[:, None] * g[Wp_idx][:, None]
    # Stage 3's parametric fit assumes Xg(t,k) is *exactly* a sum of
    # complex exponentials in t (see fit_harmonics docstring) -- the
    # outer window g(t), needed for stage 1's FFT-based leakage control,
    # actively distorts that model (amplitude tapering across a sparse,
    # non-contiguous t_idx corrupted the fit badly here -- found by
    # comparing recovered amplitudes against the dense reference and
    # seeing the right alpha locations but wrong-by-orders-of-magnitude,
    # wrongly-ranked values). Use the unwindowed CDP for the fit instead.
    Xg_rows_nowin = XT_rows * np.conj(x[Wp_idx])[:, None]

    pooled_q, pooled_w = [], []
    for ci, k in enumerate(k_idx):
        Xg_col = np.zeros(N, dtype=complex)
        Xg_col[Wp_idx] = Xg_rows[:, ci]
        freqs, coeffs, _ = be.run(Xg_col, kappa_search, seed=seed)
        q_centered = np.where(freqs < N // 2, freqs, freqs - N)
        # Convert to *global* alpha-space bin index: alpha=0 (the always-
        # dominant non-cyclic power term) sits at a DIFFERENT outer-FFT
        # bin q_centered for every channel (q_centered = -k*N/Np gives
        # alpha=0 for channel k) -- pooling raw q_centered across channels
        # without this conversion lets every channel's own DC-equivalent
        # term masquerade as a "candidate harmonic" and swamp the real
        # ones (found the hard way -- see module history/tests).
        _, alpha_ch = _to_f_alpha(k, q_centered, Np, N)
        q_global = np.round(alpha_ch * N).astype(np.int64)
        pooled_q.append(q_global)
        pooled_w.append(np.abs(coeffs))
    pooled_q = np.concatenate(pooled_q) if pooled_q else np.empty(0, dtype=int)
    pooled_w = np.concatenate(pooled_w) if pooled_w else np.empty(0)

    nonzero = pooled_q != 0
    q0, ranked = estimate_fundamental(pooled_q[nonzero], pooled_w[nonzero], q0_range)

    # Stage 3: joint refit at the known harmonic comb, reusing Wp_idx --
    # no new samples read beyond stage 1's own (a real, if secondary,
    # saving worth stating: the refit is free in sample-count terms).
    qs_comb, values = fit_harmonics(Xg_rows_nowin, Wp_idx, N, q0, n_harmonics, k_idx, Np)

    fs, alphas, vals, chans = [], [], [], []
    for ci, k in enumerate(k_idx):
        # qs_comb is already a GLOBAL alpha-space index (m*q0), unlike
        # _to_f_alpha's q_centered argument which is per-channel-relative
        # -- calling _to_f_alpha here would double-count the channel
        # offset (found the hard way; see module history/tests). alpha is
        # just qs_comb/N directly; f follows from the paper's own
        # f = f_k - alpha/2 convention (the same relationship
        # _to_f_alpha encodes, solved for this direction instead).
        fk = k / Np
        alpha_ch = qs_comb / N
        f_ch = fk - alpha_ch / 2.0
        v_ch = values[:, ci]
        if 2 * n_harmonics + 1 > kappa:
            order = np.argsort(-np.abs(v_ch))[:kappa]
            f_ch, alpha_ch, v_ch = f_ch[order], alpha_ch[order], v_ch[order]
        fs.append(f_ch); alphas.append(alpha_ch); vals.append(v_ch)
        chans.append(np.full(v_ch.size, k))

    elapsed = time.perf_counter() - t0
    f = np.concatenate(fs); alpha = np.concatenate(alphas)
    value = np.concatenate(vals); channel = np.concatenate(chans)
    return HarmonicResult(
        f, alpha, value, channel, n_positions, n_raw, elapsed, "harmonic",
        {"alpha0": q0 / N, "q0": q0, "ranked_q0": [(q, s / max(pooled_w.sum(), 1e-30))
                                                    for q, s in ranked]})


# ---------------------------------------------------------------------------
# checking
# ---------------------------------------------------------------------------
class HarmonicCheckReport(NamedTuple):
    kappa: int
    Np: int
    N: int
    dense_time: float
    harmonic_time: float
    hit_rate: float
    residual: float
    sparsity_ratio: float
    raw_sample_ratio: float
    speedup: float
    alpha0_true: float | None
    alpha0_est: float
    alpha0_rel_err: float | None
    ranked_q0: list


def check_harmonic(x, Np, kappa, q0_range, expected_data_rate=None,
                    n_harmonics=None, kappa_search_mult=3, base_backend="sfft1",
                    base_backend_kwargs=None, seed=0, a=None, g=None) -> HarmonicCheckReport:
    from s3ca import dense_ssca, _support_and_residual

    x = np.asarray(x)
    N = x.size
    if a is None or g is None:
        a_def, g_def = _default_windows(N, Np)
        a = a_def if a is None else a
        g = g_def if g is None else g

    t0 = time.perf_counter()
    S_dense, _, _ = dense_ssca(x, Np, a=a, g=g)
    dense_time = time.perf_counter() - t0

    k_idx = _centered(Np)
    q_idx = _centered(N)

    result = harmonic_ssca(x, Np, kappa, q0_range, n_harmonics=n_harmonics,
                            kappa_search_mult=kappa_search_mult,
                            base_backend=base_backend,
                            base_backend_kwargs=base_backend_kwargs,
                            seed=seed, a=a, g=g)

    hit, res = _support_and_residual(S_dense, k_idx, q_idx, result, kappa, Np, N)

    alpha0_est = result.extra["alpha0"]
    alpha0_err = (abs(alpha0_est - expected_data_rate) / expected_data_rate
                  if expected_data_rate else None)

    return HarmonicCheckReport(
        kappa=kappa, Np=Np, N=N, dense_time=dense_time, harmonic_time=result.elapsed,
        hit_rate=hit, residual=res,
        sparsity_ratio=result.n_positions / N,
        raw_sample_ratio=result.n_raw_samples_read / N,
        speedup=dense_time / result.elapsed if result.elapsed else float("inf"),
        alpha0_true=expected_data_rate, alpha0_est=alpha0_est, alpha0_rel_err=alpha0_err,
        ranked_q0=result.extra["ranked_q0"],
    )


def print_harmonic_report(r: HarmonicCheckReport) -> None:
    print(f"N={r.N}, Np={r.Np}, kappa={r.kappa}")
    print(f"{'':16}{'time (ms)':>12}{'speedup':>10}{'hit rate':>11}{'residual':>11}")
    print(f"{'dense SSCA':16}{r.dense_time*1e3:12.1f}{'--':>10}{'--':>11}{'--':>11}")
    print(f"{'harmonic':16}{r.harmonic_time*1e3:12.1f}{r.speedup:9.2f}x"
          f"{r.hit_rate:11.1%}{r.residual:11.3g}")
    print(f"evaluated channelizer/CDP at {r.sparsity_ratio:.1%} of N positions "
          f"(raw samples read: {r.raw_sample_ratio:.1%} of N)")
    if r.alpha0_true is not None:
        print(f"alpha0: true={r.alpha0_true:.8f}  estimated={r.alpha0_est:.8f}  "
              f"relative error={r.alpha0_rel_err:.2%}")
    else:
        print(f"alpha0 estimated = {r.alpha0_est:.8f} (no ground truth given)")
    print("top-ranked q0 candidates (q0, normalised score):",
          [(q, f"{s:.3f}") for q, s in r.ranked_q0])
