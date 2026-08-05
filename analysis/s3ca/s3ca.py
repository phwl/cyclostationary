"""
S3CA -- Sparse Strip Spectral Correlation Analyzer
====================================================

Implementation of

    C. J. Li, R. Rademacher, D. Boland, C. T. Jin, C. M. Spooner, P. H.W. Leong,
    "S3CA: A Sparse Strip Spectral Correlation Analyzer", IEEE SPL 2015-style
    letter (as supplied, s3ca_spl24.pdf).

built on top of a pluggable sparse-FFT backend, used strictly as a library
(unmodified):

  * "sfft1"     -- sfft_opt.py: sFFT 1.0, random hashing (Hassanieh, Indyk,
                    Katabi, Price, SODA'12).
  * "decimated" -- decimated_sfft.py: fixed decimation by D with binary
                    phase encoding, no randomness.

See `make_backend` for the pluggable-backend mechanics, and
`compare_backends` for running both on the same signal.

------------------------------------------------------------------------------
What this file provides
------------------------------------------------------------------------------
dense_ssca(x, Np, ...)
    The conventional strip spectral correlation analyzer (SSCA), computed
    densely with ordinary FFTs. This is the ground truth used for `check()`.

s3ca(x, Np, kappa, mode=..., ...)
    The sparse strip spectral correlation analyzer, in two modes:

    mode="naive"  -- Fig. 1 solid-line diagram of the paper: the Np dense
                      N-point FFTs of the SSCA are simply replaced by calls to
                      `sfft1`, one independent call per channel. The full CDP
                      matrix X_g is still computed. This is the paper's
                      "naive S3CA" (~2x speedup in the paper, Fig. 4a).

    mode="full"   -- the paper's actual S3CA: COMPIDX. The same sigma/tau
                      (Algorithm 2's Sigma, Upsilon) are used for every
                      channel's SFFT, so the set of time-domain samples any
                      channel's SFFT will ever look at is *identical* across
                      channels. That shared set W' is computed once, and the
                      channelizer + channel-data-product (CDP) -- the
                      O(N*Np*log Np) part of the SSCA -- is evaluated only at
                      the indices in W' (dilated by the channelizer's own
                      Np-sample window -- see `_channelizer_footprint` --
                      since computing X_T at time t needs a small window of
                      raw samples around t, not just x[t]), not over the
                      full length-N signal.

check(x, Np, kappa, ...)
    Runs dense_ssca and both S3CA modes on the same data and reports hit
    rate against the dense top-kappa*Np peaks, the residual L1 norm (as in
    the paper's Fig. 3(f)), timings, and the W' sparsity ratio (Fig. 4b).

------------------------------------------------------------------------------
Modeling choices / simplifications (stated explicitly)
------------------------------------------------------------------------------
* Circular convention. The paper's Eq. (1)-(2) need N + Np raw samples (Np/2
  of "halo" on each side of an N-sample block) to form X_T and X_g without
  edge effects. To keep this a clean, self-contained, single-block demo we
  treat the length-N input as one period of a periodic signal, i.e. the
  channelizer wraps circularly. Ts = 1/fs = 1 (as in the paper's
  normalization), so f_k = k/Np exactly and Delta_alpha = 1/N exactly.
  This changes nothing about the SFFT-acceleration logic (the whole point of
  the exercise); it only avoids a boundary-sample bookkeeping detail.

* Windows a(r) (channelizer taper) and g(m) (outer window) are Hamming by
  default and are shared between dense_ssca and s3ca, so comparisons between
  them are apples-to-apples regardless of which window you pick.

* What is and isn't saved, precisely. The paper claims two savings: (1) the
  channelizer/CDP need only be evaluated at |W'| time positions instead of N
  (Table I row 1: O(N*Np*log Np) -> O(|W'|*Np*log Np) -- this is a count of
  Np-point FFTs run, and is genuinely realized here: mode="full" runs |W'|
  of them, not N), and (2) the intermediate CDP matrix X'_g is |W'| x Np,
  not N x Np (paper's second bullet). Both of those are realized in
  `_channelizer_rows`/`_cdp` below: they operate on |W'|-length arrays.
  What is *not* fully realized: `sfft1`'s public API takes one dense
  length-N array, so each per-channel CDP column still gets scattered into
  an N-length (mostly zero) buffer before being handed to `sfft1` -- an
  O(N) allocation per channel, even though only |W'| of it is ever read.
  Avoiding that would mean restructuring sfft1's internals to accept a
  sparse/dict input, which is out of scope for using the library as-is, so
  it is left as a documented gap.

  Separately: the *distinct raw x[] samples* the restricted channelizer
  reads (each of the |W'| positions needs a small Np-sample window, not
  just one sample) is a different, larger quantity that can approach N even
  when |W'| is small -- see `_channelizer_footprint`. That is not one of
  the paper's two claimed savings and is reported by `check()` separately,
  for transparency, rather than folded into the headline sparsity number.
"""

from __future__ import annotations

import time
from math import gcd
from typing import NamedTuple

import numpy as np

from sfft_opt import Filter, flat_filter, sfft1
from decimated_sfft import Plan, make_plan, decode

__all__ = ["dense_ssca", "s3ca", "check", "compare_backends", "make_backend",
           "S3CAResult", "CheckReport"]


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------
def _centered(m):
    """Index range [-m//2, m - m//2) -- matches the paper's k in [-Np/2,Np/2-1)
    and q in [-N/2, N/2-1) conventions for even m."""
    return np.arange(-(m // 2), m - m // 2)


def _default_windows(N, Np):
    return np.hamming(Np), np.hamming(N)


def _to_f_alpha(k_centered, q_centered, Np, N):
    """Eq. (3)'s coordinate map, normalised to fs = 1."""
    fk = k_centered / Np
    dalpha = 1.0 / N
    alpha = fk + q_centered * dalpha
    f = (fk - q_centered * dalpha) / 2.0
    return f, alpha


# ---------------------------------------------------------------------------
# dense channelizer + CDP  (shared machinery, full-array version)
# ---------------------------------------------------------------------------
def _channelizer_dense(x, Np, a):
    """X_T(t,k) for every t in [0,N) and every channel k (centered), via one
    batched Np-point FFT per t.  O(N*Np*log Np) -- see Table I, SSCA row 1."""
    N = x.size
    k_idx = _centered(Np)
    r_off = np.arange(-(Np // 2), Np - Np // 2)             # matches k_idx pattern

    # S_mat[t, r'] = a(r) * x[(t + r) mod N],  r' indexes r_off in order
    t = np.arange(N)
    S_mat = np.empty((N, Np), dtype=complex)
    for ri, r in enumerate(r_off):
        S_mat[:, ri] = a[ri] * x[(t + r) % N]

    bracket = np.fft.fft(S_mat, axis=1)                     # fft order k' = 0..Np-1
    kprime = np.arange(Np)
    bracket *= np.exp(1j * np.pi * kprime)[None, :]         # undo r-offset phase
    order = k_idx % Np
    bracket = bracket[:, order]                              # reorder -> k_idx order

    fk = k_idx / Np
    XT = bracket * np.exp(-2j * np.pi * fk[None, :] * t[:, None])
    return XT, k_idx


def _channelizer_footprint(t_idx, N, Np):
    """The raw-signal samples the channelizer actually reads to evaluate
    X_T at every t in `t_idx`: each t needs a Np-sample window x[t-Np/2 :
    t+Np/2) (circularly), not just x[t] itself. This dilated set, not
    `t_idx` alone, is the true number of samples the restricted
    channelizer touches."""
    r_off = np.arange(-(Np // 2), Np - Np // 2)
    footprint = (np.asarray(t_idx)[:, None] + r_off[None, :]) % N
    return np.unique(footprint.ravel())


def _channelizer_rows(x, Np, a, t_idx):
    """Same as `_channelizer_dense` but only at the rows in `t_idx` --
    O(|t_idx| * Np * log Np).  This is the S3CA channelizer restriction."""
    N = x.size
    k_idx = _centered(Np)
    r_off = np.arange(-(Np // 2), Np - Np // 2)

    t_idx = np.asarray(t_idx)
    S_mat = np.empty((t_idx.size, Np), dtype=complex)
    for ri, r in enumerate(r_off):
        S_mat[:, ri] = a[ri] * x[(t_idx + r) % N]

    bracket = np.fft.fft(S_mat, axis=1)
    kprime = np.arange(Np)
    bracket *= np.exp(1j * np.pi * kprime)[None, :]
    order = k_idx % Np
    bracket = bracket[:, order]

    fk = k_idx / Np
    XT = bracket * np.exp(-2j * np.pi * fk[None, :] * t_idx[:, None])
    return XT, k_idx


# ---------------------------------------------------------------------------
# dense SSCA -- the reference / ground truth
# ---------------------------------------------------------------------------
def dense_ssca(x, Np, a=None, g=None):
    """Conventional (dense) strip spectral correlation analyzer.

    Parameters
    ----------
    x : complex array, length N.
    Np : number of channelizer bands (must divide... only needs to be even).
    a : length-Np channelizer taper (default Hamming).
    g : length-N outer window (default Hamming).

    Returns
    -------
    S : complex array, shape (N, Np).  S[qi, ki] is the SCD estimate at
        (f, alpha) = f_alpha[qi, ki, 0], f_alpha[qi, ki, 1].
    f, alpha : real arrays, shape (N, Np), the coordinate grids.
    """
    x = np.asarray(x)
    N = x.size
    if a is None or g is None:
        a_def, g_def = _default_windows(N, Np)
        a = a_def if a is None else a
        g = g_def if g is None else g

    XT, k_idx = _channelizer_dense(x, Np, a)
    Xg = XT * np.conj(x)[:, None] * g[:, None]

    Sfft = np.fft.fft(Xg, axis=0)                # bin q' = 0..N-1
    q_idx = _centered(N)
    S = Sfft[q_idx % N, :]

    f, alpha = _to_f_alpha(k_idx[None, :], q_idx[:, None], Np, N)
    return S, f, alpha


# ---------------------------------------------------------------------------
# COMPIDX equivalent: which time samples will the shared-seed SFFTs touch?
# ---------------------------------------------------------------------------
def required_time_indices(N, filt: Filter, loc_loops, est_loops, seed):
    """Reproduces exactly the sigma/tau draw and index-gather that `sfft1`
    performs internally (sfft_opt.py, the "sigmas/taus" and "idx" computation
    right before the batched FFT loop), so that calling this with the same
    (N, filt, loc_loops, est_loops, seed) that every per-channel `sfft1` call
    below will use tells us, in advance, the union of samples any of those
    calls could ever read -- without running any of them.

    This is COMPIDX (Algorithm 2) realized by reusing sfft1's own public
    seeding contract rather than reimplementing sfft1.
    """
    L = loc_loops + est_loops
    rng = np.random.default_rng(seed)
    sigmas = np.empty(L, dtype=np.int64)
    taus = np.empty(L, dtype=np.int64)
    for i in range(L):
        s = int(rng.integers(0, N))
        while gcd(s, N) != 1:
            s = int(rng.integers(0, N))
        sigmas[i] = s
        taus[i] = int(rng.integers(0, N))

    supp = np.arange(filt.Wp, dtype=np.int64)
    idx = (sigmas[:, None] * supp[None, :] + taus[:, None]) % N
    return np.unique(idx.ravel()), sigmas, taus


# ---------------------------------------------------------------------------
# Pluggable sparse-FFT backends
# ---------------------------------------------------------------------------
# S3CA's shape -- restrict the channelizer/CDP to whatever samples the
# per-channel sparse recovery will read, and share that restriction across
# all Np channels -- doesn't actually depend on *how* the per-channel sparse
# recovery decides what to read. Below are two interchangeable engines:
#
#   "sfft1"      sfft_opt.sfft1 -- random hashing (Hassanieh et al SODA'12).
#                Needs the same-seed trick (see `required_time_indices`
#                above) to make the required-sample set channel-independent;
#                left to its own devices ("naive" mode) it fragments.
#
#   "decimated"  decimated_sfft.py -- fixed decimation by D with binary phase
#                encoding. There is no randomness to share: every branch
#                reads samples at fixed residues i mod D in a small, known
#                set, so the required-sample set is *already* identical
#                across channels with no seed trick needed at all. "naive"
#                and "full" mode are therefore expected to need the same
#                samples for this backend -- itself a useful point of
#                comparison against sfft1's naive/full gap.
#
# Both expose the same two operations, `required_indices(seed)` and
# `run(x_col, kappa, seed)`, so `s3ca()` below doesn't need to know which one
# it's holding.
class _SFFT1Backend:
    name = "sfft1"

    def __init__(self, N, kappa, filt=None, B=None, loc_loops=4, est_loops=16, **kw):
        if filt is None:
            if B is None:
                B = 1 << max(1, int(round(0.5 * np.log2(max(N * kappa / 5.0, 4)))))
                B = min(B, N)
                while N % B:
                    B //= 2
            filt = flat_filter(N, B, tolerance=kw.pop("tolerance", 1e-6),
                                box_scale=kw.pop("box_scale", 1.6))
        self.N, self.filt = N, filt
        self.loc_loops, self.est_loops, self.kw = loc_loops, est_loops, kw

    def required_indices(self, seed):
        idx, _, _ = required_time_indices(self.N, self.filt, self.loc_loops,
                                           self.est_loops, seed)
        return idx

    def run(self, x_col, kappa, seed):
        freqs, coeffs = sfft1(x_col, kappa, filt=self.filt, loc_loops=self.loc_loops,
                               est_loops=self.est_loops, rng=seed, **self.kw)
        return freqs, coeffs, {}


class _DecimatedBackend:
    name = "decimated"

    def __init__(self, N, kappa, plan=None, D=None, window="kaiser", beta=12.0, **kw):
        if plan is None:
            if D is None:
                B_target = 1 << max(1, int(round(0.5 * np.log2(max(N * kappa / 5.0, 4)))))
                D = max(2, N // B_target)
                D = 1 << int(round(np.log2(D)))
                D = min(D, N // 2)
                while N % D:
                    D //= 2
            plan = make_plan(N, D, window=window, beta=beta)
        self.N, self.plan, self.kw = N, plan, kw

    def required_indices(self, seed=None):
        # deterministic: fixed residues mod D, the same for every channel and
        # every call regardless of `seed` -- accepted only for interface
        # symmetry with _SFFT1Backend.
        taus, D = self.plan.taus, self.plan.D
        return np.concatenate([np.arange(int(tau), self.N, D, dtype=np.int64)
                                for tau in taus])

    def run(self, x_col, kappa, seed=None):
        plan = self.plan
        buf = np.empty((len(plan.taus), plan.B), dtype=complex)
        for bi, tau in enumerate(plan.taus):
            buf[bi] = x_col[int(tau)::plan.D]
        Y = plan.scale * np.fft.fft(buf * plan.window, axis=-1)
        freqs, coeffs, unresolved = decode(Y, plan, **self.kw)
        if freqs.size > kappa:                    # decode() has no built-in top-k cap;
            freqs, coeffs = freqs[:kappa], coeffs[:kappa]   # already sorted by |coeff|
        return freqs, coeffs, {"unresolved_bins": int(unresolved.size)}


def make_backend(N, kappa, backend="sfft1", **kwargs):
    """Build a sparse-FFT backend for `s3ca()`. `backend` is 'sfft1' or
    'decimated'; `**kwargs` are forwarded to that backend's constructor
    (filt/B/loc_loops/est_loops/tolerance/... for sfft1; plan/D/window/beta/
    threshold/rel_tol/... for decimated)."""
    if backend == "sfft1":
        return _SFFT1Backend(N, kappa, **kwargs)
    if backend == "decimated":
        return _DecimatedBackend(N, kappa, **kwargs)
    raise ValueError(f"unknown backend {backend!r}; use 'sfft1' or 'decimated'")


# ---------------------------------------------------------------------------
# S3CA
# ---------------------------------------------------------------------------
class S3CAResult(NamedTuple):
    f: np.ndarray            # spectral frequency, one per recovered (channel, freq) pair
    alpha: np.ndarray        # cycle frequency
    value: np.ndarray        # complex SCD estimate
    channel: np.ndarray      # which channel k (centered index) each entry came from
    n_positions: int         # |W'|: how many time positions the channelizer/CDP were
                              # evaluated at, i.e. how many Np-point FFTs were run. This
                              # is the quantity that drives the paper's Table I compute
                              # and (Np-column) intermediate-storage savings; N in "naive"
                              # mode, |W'| << N in "full" mode.
    n_raw_samples_read: int  # distinct raw x[] samples actually read to do that -- >=
                              # n_positions, since each position needs a small Np-sample
                              # window around it. Not one of the paper's two claimed
                              # savings dimensions, and can approach N even when
                              # n_positions is small if W' is dense enough that the
                              # Np-windows tile over it; reported for transparency.
    elapsed: float           # wall time in seconds
    backend: str             # which sparse-FFT engine produced this ("sfft1"/"decimated")
    extra: dict              # backend-specific diagnostics, summed over channels
                              # (e.g. {"unresolved_bins": ...} for "decimated")


def s3ca(x, Np, kappa, mode="full", seed=0, a=None, g=None,
          backend="sfft1", **backend_kwargs) -> S3CAResult:
    """Sparse strip spectral correlation analyzer.

    Parameters
    ----------
    x : complex array, length N.
    Np : number of channelizer bands.
    kappa : target number of non-zero cyclic-spectrum coefficients per
        channel.
    mode : "full" (COMPIDX + shared required-sample set, restricted
        channelizer -- the paper's actual S3CA) or "naive" (dense
        channelizer/CDP, independent per-channel recovery -- the paper's
        "naive S3CA" baseline).
    seed : rng seed, used by the "sfft1" backend (ignored by "decimated",
        which is deterministic). In "full" mode this seed is reused,
        unchanged, for every channel's sparse-recovery call -- that reuse is
        exactly what makes the required-sample set identical across channels
        for a randomized backend (see module docstring and
        `required_time_indices`); a deterministic backend gets this for
        free.
    backend : "sfft1" (sfft_opt.sfft1, random hashing) or "decimated"
        (decimated_sfft.py, fixed decimation) -- or an already-constructed
        backend object from `make_backend`, to reuse across repeated calls
        (e.g. from `check()`) instead of rebuilding it (its Filter / Plan)
        every time.
    **backend_kwargs : forwarded to `make_backend` if `backend` is a string.

    Returns
    -------
    S3CAResult
    """
    x = np.asarray(x)
    N = x.size
    if a is None or g is None:
        a_def, g_def = _default_windows(N, Np)
        a = a_def if a is None else a
        g = g_def if g is None else g

    be = backend if not isinstance(backend, str) else make_backend(
        N, kappa, backend=backend, **backend_kwargs)

    k_idx = _centered(Np)
    t0 = time.perf_counter()

    fs, alphas, values, chans = [], [], [], []
    extra_total: dict = {}

    def _accumulate(extra):
        for key, val in extra.items():
            extra_total[key] = extra_total.get(key, 0) + val

    if mode == "naive":
        # Dense channelizer/CDP (no savings there), independent per-channel
        # recovery (no savings on the shared-requirement trick either --
        # for "sfft1" this means a fresh seed per channel; "decimated" has
        # no seed to vary, so it degenerates to the same thing as "full").
        XT, _ = _channelizer_dense(x, Np, a)
        Xg = XT * np.conj(x)[:, None] * g[:, None]
        n_positions = n_raw = N
        for ci, k in enumerate(k_idx):
            freqs, coeffs, extra = be.run(Xg[:, ci], kappa,
                                           seed=None if seed is None else seed + ci)
            _accumulate(extra)
            q_centered = np.where(freqs < N // 2, freqs, freqs - N)
            f, alpha = _to_f_alpha(k, q_centered, Np, N)
            fs.append(f); alphas.append(alpha); values.append(coeffs)
            chans.append(np.full(freqs.size, k))

    elif mode == "full":
        # COMPIDX: the required-sample set is identical across channels
        # (by the same-seed trick for "sfft1", or inherently for
        # "decimated"). Compute it once, then only fill in the channelizer +
        # CDP at those samples.
        Wp_idx = be.required_indices(seed)
        n_positions = Wp_idx.size
        n_raw = _channelizer_footprint(Wp_idx, N, Np).size

        XT_rows, _ = _channelizer_rows(x, Np, a, Wp_idx)
        Xg_rows = XT_rows * np.conj(x[Wp_idx])[:, None] * g[Wp_idx][:, None]

        for ci, k in enumerate(k_idx):
            Xg_col = np.zeros(N, dtype=complex)
            Xg_col[Wp_idx] = Xg_rows[:, ci]
            freqs, coeffs, extra = be.run(Xg_col, kappa, seed=seed)
            _accumulate(extra)
            q_centered = np.where(freqs < N // 2, freqs, freqs - N)
            f, alpha = _to_f_alpha(k, q_centered, Np, N)
            fs.append(f); alphas.append(alpha); values.append(coeffs)
            chans.append(np.full(freqs.size, k))
    else:
        raise ValueError(f"mode must be 'naive' or 'full', got {mode!r}")

    elapsed = time.perf_counter() - t0
    f = np.concatenate(fs) if fs else np.empty(0)
    alpha = np.concatenate(alphas) if alphas else np.empty(0)
    value = np.concatenate(values) if values else np.empty(0, dtype=complex)
    channel = np.concatenate(chans) if chans else np.empty(0, dtype=int)
    return S3CAResult(f, alpha, value, channel, n_positions, n_raw, elapsed,
                       be.name, extra_total)


# ---------------------------------------------------------------------------
# checking: compare S3CA against the dense SSCA it is approximating
# ---------------------------------------------------------------------------
class CheckReport(NamedTuple):
    backend: str
    kappa: int
    Np: int
    N: int
    dense_time: float
    naive_time: float
    full_time: float
    naive_hit_rate: float
    full_hit_rate: float
    naive_residual: float          # ||S_dense - S_naive||_1 / (kappa*Np), cf. paper Fig. 3(f)
    full_residual: float
    full_sparsity_ratio: float          # |W'| / N: fraction of channelizer *positions*
                                         # evaluated -- the paper's Table I / Fig. 4(b) metric
    full_raw_sample_ratio: float        # distinct raw x[] samples read / N (see
                                         # S3CAResult.n_raw_samples_read -- not the same
                                         # quantity, reported for transparency)
    naive_style_sparsity_ratio: float   # what |union W'_k| / N would be if every
                                         # channel independently decided what it needs
    naive_speedup: float
    full_speedup: float
    alpha_grid_hit_rate: float | None   # optional: fraction of recovered alphas
                                         # near a known m * data_rate grid
    fill_rate: float              # (channels' total recovered coefficients) / (kappa*Np)
                                   # for "full". sfft1 always returns close to kappa per
                                   # channel; "decimated"'s singleton test can refuse to
                                   # report far fewer than kappa if the per-channel CDP
                                   # isn't genuinely kappa-sparse -- a low fill_rate, not a
                                   # low hit_rate, is usually why its accuracy looks worse:
                                   # it is abstaining, not guessing wrong.
    full_extra: dict             # backend-specific diagnostics from the "full" run
                                  # (e.g. {"unresolved_bins": ...} for "decimated")


def _support_and_residual(S_dense, k_idx, q_idx, result: S3CAResult, kappa, Np, N):
    """Compare a S3CAResult against the dense reference on the same (Np,N)
    coordinate grid.

    `sfft1` promises to recover, *per call*, the top-kappa coefficients of
    whatever length-N signal it was given -- and s3ca() calls it once per
    channel. So the fair per-channel target is the top-kappa dense bins of
    that same channel, not some globally chosen top-kappa*Np set (which,
    for a real signal whose energy is unevenly spread across channels, can
    be dominated by a handful of channels and unfairly penalise a
    per-channel recovery scheme). Total hit rate is summed over channels;
    the L1 residual keeps the paper's Fig. 3(f) normalisation.
    """
    mag = np.abs(S_dense)
    k_pos = {int(k): i for i, k in enumerate(k_idx)}
    dalpha = 1.0 / N

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
            continue                      # shouldn't happen, but be defensive
        recon[qi, ki] = val
        if qi in true_sets[ki]:
            hit += 1
    hit_rate = hit / n_true if n_true else float("nan")
    residual = np.sum(np.abs(S_dense - recon)) / (kappa * Np)
    return hit_rate, residual


def check(x, Np, kappa, a=None, g=None, backend="sfft1", seed=0,
          expected_data_rate=None, alpha_tol_bins=1, **backend_kwargs) -> CheckReport:
    """Run dense_ssca and both S3CA modes on the same `x` and report
    hit rate / residual / timing / sparsity, mirroring the paper's own
    verification methodology (Fig. 3's residual, Fig. 4's speedup and
    sparsity plots).

    `backend` is "sfft1", "decimated", or an already-built backend object
    from `make_backend` (built once and reused, so naive/full comparisons
    and the diagnostic loop below all see the identical filter/plan).

    If `expected_data_rate` is given (a cycle frequency, in the same units as
    alpha, i.e. normalised to fs=1), also reports what fraction of S3CA's
    (mode="full") recovered peaks land within `alpha_tol_bins` * (1/N) of an
    integer multiple of it -- a check against independently-known physics,
    not just against the dense baseline.
    """
    x = np.asarray(x)
    N = x.size
    if a is None or g is None:
        a_def, g_def = _default_windows(N, Np)
        a = a_def if a is None else a
        g = g_def if g is None else g

    be = backend if not isinstance(backend, str) else make_backend(
        N, kappa, backend=backend, **backend_kwargs)

    t0 = time.perf_counter()
    S_dense, _, _ = dense_ssca(x, Np, a=a, g=g)
    dense_time = time.perf_counter() - t0

    k_idx = _centered(Np)
    q_idx = _centered(N)

    naive = s3ca(x, Np, kappa, mode="naive", seed=seed, a=a, g=g, backend=be)
    full = s3ca(x, Np, kappa, mode="full", seed=seed, a=a, g=g, backend=be)

    naive_hit, naive_res = _support_and_residual(S_dense, k_idx, q_idx, naive, kappa, Np, N)
    full_hit, full_res = _support_and_residual(S_dense, k_idx, q_idx, full, kappa, Np, N)

    # Diagnostic: this is *why* naive S3CA barely helps for a randomized
    # backend. If every channel decided what it needs independently (as
    # "naive" does), the union of required samples across Np independent
    # decisions grows with Np for "sfft1" -- but not for "decimated", whose
    # required set is fixed regardless of seed.
    union = set()
    for ci in range(Np):
        idx_ci = be.required_indices(None if seed is None else seed + ci)
        union.update(idx_ci.tolist())
    naive_style_ratio = len(union) / N

    alpha_grid_hit = None
    if expected_data_rate is not None and full.alpha.size:
        m = np.round(full.alpha / expected_data_rate)
        nearest = m * expected_data_rate
        tol = alpha_tol_bins / N
        alpha_grid_hit = float(np.mean(np.abs(full.alpha - nearest) <= tol))

    return CheckReport(
        backend=be.name,
        kappa=kappa, Np=Np, N=N,
        dense_time=dense_time, naive_time=naive.elapsed, full_time=full.elapsed,
        naive_hit_rate=naive_hit, full_hit_rate=full_hit,
        naive_residual=naive_res, full_residual=full_res,
        full_sparsity_ratio=full.n_positions / N,
        full_raw_sample_ratio=full.n_raw_samples_read / N,
        naive_style_sparsity_ratio=naive_style_ratio,
        naive_speedup=dense_time / naive.elapsed if naive.elapsed else float("inf"),
        full_speedup=dense_time / full.elapsed if full.elapsed else float("inf"),
        alpha_grid_hit_rate=alpha_grid_hit,
        fill_rate=full.channel.size / (kappa * Np),
        full_extra=full.extra,
    )


def compare_backends(x, Np, kappa, backends=("sfft1", "decimated"),
                      backend_kwargs=None, **common_kw) -> dict:
    """Run `check()` once per backend on the *same* signal and return
    {backend_name: CheckReport}, for a direct accuracy/speed comparison of
    S3CA built on different sparse-FFT engines.

    `backend_kwargs`, if given, is ``{backend_name: {...}}`` -- per-backend
    construction kwargs (sfft1 takes loc_loops/est_loops/tolerance/B/filt/...;
    decimated takes D/window/beta/threshold/rel_tol/...), since the two
    engines don't share a parameter vocabulary. `**common_kw` (seed,
    expected_data_rate, alpha_tol_bins, a, g) is passed to every `check()`
    call unchanged.
    """
    backend_kwargs = backend_kwargs or {}
    return {b: check(x, Np, kappa, backend=b, **common_kw, **backend_kwargs.get(b, {}))
            for b in backends}


def print_report(r: CheckReport) -> None:
    print(f"backend = {r.backend!r}, N = 2^{np.log2(r.N):.0f} ({r.N} samples), "
          f"Np = {r.Np}, kappa = {r.kappa}")
    print(f"{'':16}{'time (ms)':>12}{'speedup':>10}{'hit rate':>11}{'residual':>11}")
    print(f"{'dense SSCA':16}{r.dense_time*1e3:12.1f}{'--':>10}{'--':>11}{'--':>11}")
    print(f"{'naive S3CA':16}{r.naive_time*1e3:12.1f}{r.naive_speedup:9.2f}x"
          f"{r.naive_hit_rate:11.1%}{r.naive_residual:11.3g}")
    print(f"{'S3CA (full)':16}{r.full_time*1e3:12.1f}{r.full_speedup:9.2f}x"
          f"{r.full_hit_rate:11.1%}{r.full_residual:11.3g}")
    print(f"S3CA (full) evaluated the channelizer/CDP at {r.full_sparsity_ratio:.1%} of "
          f"the N time positions (|W'|/N -- the paper's Table I / Fig. 4b metric: this "
          f"is also the intermediate-matrix-size ratio, {r.full_sparsity_ratio:.1%} x Np "
          f"instead of N x Np)")
    print(f"  -- vs. {r.naive_style_sparsity_ratio:.1%} of positions that Np={r.Np} "
          f"independently-deciding channels would need "
          f"({'no COMPIDX gap for a deterministic backend' if abs(r.naive_style_sparsity_ratio - r.full_sparsity_ratio) < 1e-9 else 'why naive S3CA barely helps here'})")
    print(f"  (distinct raw x[] samples read to do that: {r.full_raw_sample_ratio:.1%} of N "
          f"-- larger than |W'|/N since each position needs a small Np-sample window "
          f"around it; not one of the paper's claimed savings, shown for transparency)")
    if r.alpha_grid_hit_rate is not None:
        print(f"S3CA (full) recovered peaks landing on the expected cycle-frequency "
              f"grid: {r.alpha_grid_hit_rate:.1%}")
    if r.full_extra:
        print(f"S3CA (full) backend diagnostics: {r.full_extra}")


def print_comparison(reports: dict) -> None:
    """Side-by-side accuracy/speed table for `compare_backends()`'s output."""
    names = list(reports)
    r0 = reports[names[0]]
    namew = max(12, max(len(n) for n in names) + 1)
    print(f"N = 2^{np.log2(r0.N):.0f} ({r0.N} samples), Np = {r0.Np}, kappa = {r0.kappa}"
          f"   (dense SSCA reference: {r0.dense_time*1e3:.1f} ms)")
    wcol = "|W'|/N"
    header = (f"{'backend':{namew}}{'hit rate':>10}{'residual':>11}{'speedup':>9}"
              f"{wcol:>9}{'fill rate':>11}{'alpha grid':>12}")
    print(header)
    for name in names:
        r = reports[name]
        agr = f"{r.alpha_grid_hit_rate:.1%}" if r.alpha_grid_hit_rate is not None else "--"
        print(f"{name:{namew}}{r.full_hit_rate:10.1%}{r.full_residual:11.3g}"
              f"{r.full_speedup:8.2f}x{r.full_sparsity_ratio:9.1%}{r.fill_rate:11.1%}{agr:>12}")
        if r.full_extra:
            print(f"{'':{namew}}{r.full_extra}")
    if any(reports[n].fill_rate < 0.5 for n in names):
        print("(fill rate = recovered coefficients / (kappa*Np): well below 100% means "
              "that backend is finding fewer than kappa per channel -- usually because "
              "the per-channel CDP isn't genuinely kappa-sparse and singleton/collision "
              "tests are (correctly) abstaining rather than guessing. A low fill rate, "
              "not a low hit rate, is usually the real story -- see full_extra above.)")
