"""
debias.py -- backend-agnostic debiasing and iterative refinement for S3CA
sparse-FFT backends.

------------------------------------------------------------------------------
The idea
------------------------------------------------------------------------------
Every backend (sfft1, decimated, rffast) estimates each recovered
coefficient's *value* more or less independently -- sfft1 by a
per-candidate median over loops, rffast by a per-bin single-frequency fit,
etc. When several genuine coefficients are close together, or when a strong
one leaks into the estimate of a weak one, those independent estimates are
biased by cross-talk: each is computed as if the others weren't there.

Two cheap, sample-free fixes, both operating only on samples the backend
ALREADY read (no new channelizer/CDP evaluations):

1. **Debiasing (joint refit).** Take the backend's recovered *support*
   (the set of frequency locations) as given, and re-estimate ALL their
   values *jointly* with a single least-squares solve against the already-
   read samples. This deconflicts cross-talk: each coefficient is fit in
   the presence of the others rather than as if it were alone. This is the
   standard "debiasing" step from compressive sensing (e.g. the least-
   squares re-solve on the recovered support after OMP/CoSaMP/L1).

2. **Iterative refinement (peel + re-search).** After debiasing, subtract
   the jointly-fitted model from the samples and hand the *residual* back
   to the backend to search again. Coefficients the first pass missed --
   because a stronger neighbour masked them in the backend's own internal
   scoring -- can surface in the residual once the strong ones are removed.
   Add any new finds to the support and repeat the joint refit. This is
   CoSaMP/OMP-style support growth, wrapped around an arbitrary sparse-FFT
   backend rather than a dense sensing matrix.

Directly targets the two failure modes diagnosed earlier in this project:
cross-talk between simultaneously-recovered frequencies (fixed by 1), and
real coefficients a backend's blind search missed under masking (fixed by
2). Unlike harmonic.py, this assumes NO structure in the support -- it
works on any signal, which is the point: it's a general refinement any
backend gets for free, not a cyclostationary-specific model.

------------------------------------------------------------------------------
What is and isn't reused from the backend
------------------------------------------------------------------------------
* The SUPPORT (frequency locations) comes from the backend. Debiasing
  trusts the backend to find *where* the coefficients are and only
  re-estimates *what* they are; residual re-search then lets the backend
  extend the support too.
* The SAMPLES are whatever the backend read (`required_indices`). The
  joint fit and the residual are both formed only on those indices, so no
  new channelizer/CDP work is done -- the refinement is free in
  sample-count terms (it costs a kappa-sized dense linear solve per
  channel, which is cheap relative to the channelizer for kappa << N).

------------------------------------------------------------------------------
Honest limitations
------------------------------------------------------------------------------
* Conditioning. If the backend's support has two locations closer than the
  sample set can resolve, the joint least-squares system is ill-conditioned
  and the refit can amplify noise rather than reduce cross-talk. `rcond`
  guards against outright blow-up (numpy lstsq drops near-singular
  directions), but a badly-collided support is a garbage-in situation the
  refit can't rescue.
* It cannot invent resolution the samples don't contain. If the backend
  never read enough distinct samples to separate two frequencies, no
  amount of joint fitting will separate them -- residual re-search can only
  find coefficients that ARE separable given the samples on hand.
* Residual re-search re-runs the backend, so its cost scales with the
  number of refinement iterations. Default is a small fixed budget.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

from s3ca import (_centered, _default_windows, _channelizer_rows,
                   _channelizer_footprint, _to_f_alpha, make_backend)

__all__ = ["debiased_s3ca", "check_debiased",
           "DebiasedCheckReport", "print_debiased_report"]


# ---------------------------------------------------------------------------
# core: joint refit of one channel's support against its read samples
# ---------------------------------------------------------------------------
def _fit_support(Xg_at_idx, t_idx, N, support_q, rcond=None):
    """Joint least-squares estimate of the coefficients at frequency bins
    `support_q` (integer, centered indices allowed) from samples
    Xg_at_idx read at time positions `t_idx`.

    Model: Xg(t) = (1/N) * sum_{q in support} X[q] * exp(+i 2pi t q / N),
    matching numpy's fft/ifft convention (Sfft = fft(Xg), so the inverse
    that expresses Xg in terms of Sfft carries the +sign and the 1/N --
    the same convention derivation as harmonic.fit_harmonics, kept
    consistent here). Returns X[q] on dense_ssca's raw-fft scale (no 1/N),
    so the factor of N is folded back in.
    """
    if support_q.size == 0:
        return np.empty(0, dtype=complex)
    A = np.exp(1j * 2 * np.pi * np.outer(t_idx, support_q) / N)
    coeffs, *_ = np.linalg.lstsq(A, Xg_at_idx, rcond=rcond)
    return coeffs * N


# ---------------------------------------------------------------------------
# full driver -- mirrors s3ca.s3ca(mode="full"), plus refinement
# ---------------------------------------------------------------------------
class DebiasedResult(NamedTuple):
    f: np.ndarray
    alpha: np.ndarray
    value: np.ndarray
    channel: np.ndarray
    n_positions: int
    n_raw_samples_read: int
    elapsed: float
    backend: str
    extra: dict


def debiased_s3ca(x, Np, kappa, base_backend="sfft1", base_backend_kwargs=None,
                   refine_iters=2, seed=0, a=None, g=None, rcond=None) -> DebiasedResult:
    """S3CA (COMPIDX / mode="full" shape) with debiasing + iterative
    refinement wrapped around any base backend -- see module docstring.
    """
    x = np.asarray(x)
    N = x.size
    if a is None or g is None:
        a_def, g_def = _default_windows(N, Np)
        a = a_def if a is None else a
        g = g_def if g is None else g
    base_backend_kwargs = base_backend_kwargs or {}

    be = make_backend(N, kappa, backend=base_backend, **base_backend_kwargs)
    t0 = time.perf_counter()
    k_idx = _centered(Np)

    Wp_idx = be.required_indices(seed)
    n_positions = Wp_idx.size
    n_raw = _channelizer_footprint(Wp_idx, N, Np).size

    XT_rows, _ = _channelizer_rows(x, Np, a, Wp_idx)
    # Windowed CDP for the backend's own recovery (it expects the same
    # windowed data dense_ssca/s3ca feed it); unwindowed CDP for the joint
    # least-squares fit (which assumes a pure exponential-sum model -- the
    # outer window would distort it, the same subtlety hit in harmonic.py).
    Xg_win = XT_rows * np.conj(x[Wp_idx])[:, None] * g[Wp_idx][:, None]
    Xg_nowin = XT_rows * np.conj(x[Wp_idx])[:, None]

    fs, alphas, vals, chans = [], [], [], []
    for ci, k in enumerate(k_idx):
        # Support search uses the windowed CDP (what the backend expects);
        # the joint refit uses the unwindowed CDP (what the model assumes).
        # We drive both through debias_channel by giving it the windowed
        # data for the backend calls but refitting on unwindowed -- so
        # debias_channel needs both. Simpler: run the backend's support
        # search here on the windowed column, then refit on unwindowed.
        buf = np.zeros(N, dtype=complex)
        buf[Wp_idx] = Xg_win[:, ci]
        freqs, _, _ = be.run(buf, kappa, seed=seed)
        support = set(int(q) for q in freqs)

        def refit(support_set):
            sq = np.array(sorted(support_set), dtype=np.int64) % N
            c = _fit_support(Xg_nowin[:, ci], Wp_idx, N, sq, rcond=rcond)
            sq_centered = np.where(sq < N // 2, sq, sq - N)
            return sq, sq_centered, c

        support_mod, support_centered, coeffs = refit(support)
        prev_resid_norm = None
        for _ in range(refine_iters):
            model = (np.exp(1j * 2 * np.pi * np.outer(Wp_idx, support_mod) / N)
                     @ (coeffs / N))
            resid = Xg_nowin[:, ci] - model
            resid_norm = np.linalg.norm(resid)
            # Stop if the residual has stopped shrinking meaningfully -- at
            # that point what's left is noise, and re-searching it just
            # injects spurious candidates that (since the backend always
            # returns a full kappa) crowd out real ones in the top-kappa
            # cap. This guard is essential: without it, residual re-search
            # on a well-fit channel actively HURTS (found directly -- the
            # unguarded version added ~kappa new candidates per channel,
            # none of them real, and dropped hit rate ~10 points).
            if prev_resid_norm is not None and resid_norm > prev_resid_norm * 0.9:
                break
            prev_resid_norm = resid_norm
            buf[:] = 0
            buf[Wp_idx] = resid * g[Wp_idx]
            new_freqs, _, _ = be.run(buf, kappa, seed=seed)
            # Only admit a new candidate if, once jointly refit alongside
            # the existing support, its own amplitude clears a noise-floor
            # threshold tied to the current residual -- a candidate that
            # can't beat the residual RMS is not distinguishable from noise.
            cand_support = support | set(int(q) for q in new_freqs)
            if cand_support == support:
                break
            cq = np.array(sorted(cand_support), dtype=np.int64) % N
            cc = _fit_support(Xg_nowin[:, ci], Wp_idx, N, cq, rcond=rcond)
            cq_centered = np.where(cq < N // 2, cq, cq - N)
            noise_floor = resid_norm / np.sqrt(max(Wp_idx.size, 1)) * N
            new_support = support | {int(q) for q, amp in zip(cq_centered, cc)
                                     if abs(amp) > noise_floor and int(q) not in support}
            if new_support == support:
                break
            support = new_support
            support_mod, support_centered, coeffs = refit(support)

        if support_centered.size > kappa:
            keep = np.argsort(-np.abs(coeffs))[:kappa]
            support_centered, coeffs = support_centered[keep], coeffs[keep]

        f_ch, alpha_ch = _to_f_alpha(k, support_centered, Np, N)
        fs.append(f_ch); alphas.append(alpha_ch); vals.append(coeffs)
        chans.append(np.full(coeffs.size, k))

    elapsed = time.perf_counter() - t0
    f = np.concatenate(fs) if fs else np.empty(0)
    alpha = np.concatenate(alphas) if alphas else np.empty(0)
    value = np.concatenate(vals) if vals else np.empty(0, dtype=complex)
    channel = np.concatenate(chans) if chans else np.empty(0, dtype=int)
    return DebiasedResult(f, alpha, value, channel, n_positions, n_raw, elapsed,
                           f"debiased-{be.name}", {"refine_iters": refine_iters})


# ---------------------------------------------------------------------------
# checking
# ---------------------------------------------------------------------------
class DebiasedCheckReport(NamedTuple):
    backend: str
    kappa: int
    Np: int
    N: int
    dense_time: float
    base_time: float
    debiased_time: float
    base_hit_rate: float
    debiased_hit_rate: float
    base_residual: float
    debiased_residual: float
    sparsity_ratio: float
    speedup: float
    alpha_grid_hit_rate: float | None


def check_debiased(x, Np, kappa, base_backend="sfft1", base_backend_kwargs=None,
                    refine_iters=2, seed=0, a=None, g=None,
                    expected_data_rate=None, alpha_tol_bins=1) -> DebiasedCheckReport:
    """Compare a base backend against its debiased+refined version on the
    same signal, both against the dense SSCA reference."""
    from s3ca import dense_ssca, _support_and_residual, s3ca

    x = np.asarray(x)
    N = x.size
    if a is None or g is None:
        a_def, g_def = _default_windows(N, Np)
        a = a_def if a is None else a
        g = g_def if g is None else g
    base_backend_kwargs = base_backend_kwargs or {}

    t0 = time.perf_counter()
    S_dense, _, _ = dense_ssca(x, Np, a=a, g=g)
    dense_time = time.perf_counter() - t0

    k_idx = _centered(Np)
    q_idx = _centered(N)

    # base backend, mode="full", no refinement
    base = s3ca(x, Np, kappa, mode="full", seed=seed, a=a, g=g,
                backend=base_backend, **base_backend_kwargs)
    base_hit, base_res = _support_and_residual(S_dense, k_idx, q_idx, base, kappa, Np, N)

    # debiased + refined
    deb = debiased_s3ca(x, Np, kappa, base_backend=base_backend,
                        base_backend_kwargs=base_backend_kwargs,
                        refine_iters=refine_iters, seed=seed, a=a, g=g)
    deb_hit, deb_res = _support_and_residual(S_dense, k_idx, q_idx, deb, kappa, Np, N)

    alpha_grid_hit = None
    if expected_data_rate is not None and deb.alpha.size:
        m = np.round(deb.alpha / expected_data_rate)
        nearest = m * expected_data_rate
        tol = alpha_tol_bins / N
        alpha_grid_hit = float(np.mean(np.abs(deb.alpha - nearest) <= tol))

    return DebiasedCheckReport(
        backend=base_backend, kappa=kappa, Np=Np, N=N,
        dense_time=dense_time, base_time=base.elapsed, debiased_time=deb.elapsed,
        base_hit_rate=base_hit, debiased_hit_rate=deb_hit,
        base_residual=base_res, debiased_residual=deb_res,
        sparsity_ratio=deb.n_positions / N,
        speedup=dense_time / deb.elapsed if deb.elapsed else float("inf"),
        alpha_grid_hit_rate=alpha_grid_hit,
    )


def print_debiased_report(r: DebiasedCheckReport) -> None:
    print(f"base backend = {r.backend!r}, N = {r.N}, Np = {r.Np}, kappa = {r.kappa}")
    print(f"{'':22}{'time (ms)':>12}{'hit rate':>11}{'residual':>11}")
    print(f"{'dense SSCA':22}{r.dense_time*1e3:12.1f}{'--':>11}{'--':>11}")
    print(f"{'base ('+r.backend+')':22}{r.base_time*1e3:12.1f}"
          f"{r.base_hit_rate:11.1%}{r.base_residual:11.3g}")
    print(f"{'debiased + refined':22}{r.debiased_time*1e3:12.1f}"
          f"{r.debiased_hit_rate:11.1%}{r.debiased_residual:11.3g}")
    delta = r.debiased_hit_rate - r.base_hit_rate
    print(f"hit-rate change from refinement: {delta:+.1%}")
    if r.alpha_grid_hit_rate is not None:
        print(f"debiased peaks on expected alpha grid: {r.alpha_grid_hit_rate:.1%}")
