"""
alpha_profile.py -- estimate the complete ALPHA PROFILE of the spectral
correlation density, cheaply, by combining the parametric cycle-frequency
estimate with a targeted evaluation on the harmonic comb.

------------------------------------------------------------------------------
What the alpha profile is, and why it is its own problem
------------------------------------------------------------------------------
The alpha profile is
    P(alpha) = max_f |S_X^alpha(f)|
-- the peak of the SCD over spectral frequency f at each cycle frequency
alpha (the lower panels of Fig. 3 in the S3CA paper). It is a 1-D summary of
the 2-D SCD that answers "how much cyclic correlation exists at cycle
frequency alpha", which is what most detection/classification actually uses.

Three observations make this a *different, easier* target than either
full-SCD reconstruction or single-alpha0 detection:

1. The profile is FAR sparser than the SCD surface. Measured on DSSS-BPSK,
   only ~8% of alpha bins are meaningfully nonzero, and its entire support
   is the harmonic comb {m*alpha0}. (The full surface has a heavy tail in f
   *and* alpha; the profile collapses the f axis away.)
2. alpha0 is exactly what the parametric detector (cyclic_detect.py)
   estimates cheaply and accurately from a small subsample.
3. Once alpha0 is known, the profile VALUE at each harmonic m*alpha0 is a
   single-cycle-frequency evaluation: P(m*alpha0) = max_k |DTFT_t[X_g(t,k)]
   at that cycle frequency|. No 2-D search, no sparse recovery, no
   exact-sparsity assumption -- and it is immune to the approximate-
   sparsity that defeats reconstruction, because we never represent the
   tail.

So the recipe is: (a) estimate alpha0 parametrically, (b) evaluate the
profile on the comb {m*alpha0 : m = -M..M}. This gives the complete profile
(every populated alpha) at a fraction of the cost of reconstructing the
surface, and -- unlike single-alpha0 detection -- it returns the full set of
harmonic magnitudes, not just "a feature is present".

------------------------------------------------------------------------------
The one subtlety that matters (and bit every method in this project)
------------------------------------------------------------------------------
The channelizer output X_T(t,k) is DOWN-CONVERTED: it already carries a
factor exp(-i 2 pi f_k t) with f_k = k/Np. So in dense_ssca the SCD relation
is alpha = f_k + q/N, i.e. a global cycle frequency alpha maps to the
outer-FFT frequency (alpha - f_k) FOR CHANNEL k, not alpha itself. Evaluating
every channel's DTFT at the same global alpha (the naive thing) is wrong and
produces large, systematic errors on the weaker harmonics. `profile_at`
evaluates channel k at (alpha - f_k), which reproduces dense_ssca exactly.

------------------------------------------------------------------------------
Limitations
------------------------------------------------------------------------------
* Assumes a single fundamental alpha0 (a harmonic comb). Multiple
  independent cyclic features would need one comb per fundamental (estimate,
  evaluate, optionally peel, repeat).
* Returns the profile ON the comb. If you want the profile at cycle
  frequencies OFF the comb (e.g. to confirm they are empty, or for signals
  whose cyclic features are not a single comb), pass an explicit alpha grid
  to `profile_on_grid` instead -- that costs O(n_alpha * n_samples * Np),
  the same scan cost as cyclic_detect.
* The windowed CDP (matching dense_ssca) is used so magnitudes match the
  paper's convention; see the note in `profile_at`.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

from s3ca import (_centered, _default_windows, _channelizer_rows, dense_ssca)
from cyclic_detect import estimate_alpha0

__all__ = ["profile_at", "profile_on_grid", "harmonic_alpha_profile",
           "dense_alpha_profile", "AlphaProfileResult"]


# ---------------------------------------------------------------------------
# core evaluation of the profile at given cycle frequencies
# ---------------------------------------------------------------------------
def profile_at(Xg, t_idx, N, Np, alphas, k_idx=None):
    """Evaluate P(alpha) = max_k |DTFT_t X_g(t,k) at cycle freq alpha| for
    each alpha, using the (windowed) CDP rows Xg (shape (len(t_idx), Np)).

    Channel k is evaluated at the DOWN-CONVERTED frequency (alpha - f_k),
    f_k = k/Np, because the channelizer already applied exp(-i 2 pi f_k t)
    -- see module docstring. This is what makes the magnitudes match
    dense_ssca; evaluating at the raw global alpha instead gives large
    systematic errors on weak harmonics.

    Returns (profile, argmax_channel): profile[j] = P(alphas[j]),
    argmax_channel[j] = the channel k achieving that max (i.e. where in
    spectral frequency the peak sits).
    """
    if k_idx is None:
        k_idx = _centered(Np)
    fk = k_idx / Np                                    # (Np,)
    M = t_idx.size
    alphas = np.asarray(alphas)
    # Vectorized over (alpha, channel): for each alpha and channel the outer
    # frequency is (alpha - f_k). Build the full (n_alpha, Np) magnitude
    # table via einsum over the shared sample axis, in modest chunks to
    # bound the (n_alpha*Np, M) phase memory.
    prof = np.empty(alphas.size)
    argk = np.empty(alphas.size, dtype=int)
    max_elems = 40_000_000
    chunk = max(1, int(max_elems // (max(M, 1) * Np)))
    for s in range(0, alphas.size, chunk):
        ac = alphas[s:s + chunk]                        # (c,)
        freq = ac[:, None] - fk[None, :]                # (c, Np)
        # phase[c,k,t] = exp(-2i pi freq[c,k] t); contract with Xg[t,k]
        phase = np.exp(-2j * np.pi * freq[:, :, None] * t_idx[None, None, :])
        vals = np.abs(np.einsum('ckt,tk->ck', phase, Xg)) / M * N   # (c, Np)
        argk[s:s + chunk] = np.argmax(vals, axis=1)
        prof[s:s + chunk] = vals[np.arange(vals.shape[0]), argk[s:s + chunk]]
    return prof, k_idx[argk]


def profile_on_grid(x, Np, alphas, idx=None, a=None, g=None):
    """Profile over an arbitrary cycle-frequency grid `alphas`, from samples
    at `idx` (all samples if None). Cost ~ O(len(alphas) * len(idx) * Np).
    Use `harmonic_alpha_profile` instead when the profile is a single comb
    (far cheaper: only the comb points are evaluated)."""
    x = np.asarray(x); N = x.size
    if a is None or g is None:
        a_d, g_d = _default_windows(N, Np)
        a = a_d if a is None else a
        g = g_d if g is None else g
    if idx is None:
        idx = np.arange(N)
    XT, _ = _channelizer_rows(x, Np, a, idx)
    Xg = XT * np.conj(x[idx])[:, None] * g[idx][:, None]
    return profile_at(Xg, idx, N, Np, alphas)


# ---------------------------------------------------------------------------
# dense reference profile
# ---------------------------------------------------------------------------
def dense_alpha_profile(x, Np, n_bins=2000, a=None, g=None):
    """Ground-truth alpha profile from the full dense SSCA: bin |SCD| onto an
    alpha grid and max-reduce over spectral frequency. Returns
    (alpha_centers, profile)."""
    S, f, alpha = dense_ssca(x, Np, a=a, g=g)
    mag = np.abs(S).ravel()
    al = alpha.ravel()
    edges = np.linspace(-1, 1, n_bins + 1)
    prof = np.zeros(n_bins)
    idx = np.clip(np.searchsorted(edges, al) - 1, 0, n_bins - 1)
    np.maximum.at(prof, idx, mag)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prof


# ---------------------------------------------------------------------------
# the cheap harmonic-comb profile
# ---------------------------------------------------------------------------
class AlphaProfileResult(NamedTuple):
    alphas: np.ndarray          # cycle frequencies evaluated (the comb)
    profile: np.ndarray         # P(alpha) at each
    peak_channels: np.ndarray   # channel (spectral-freq bin) achieving each max
    alpha0: float               # estimated fundamental
    n_samples_used: int
    elapsed: float


def harmonic_alpha_profile(x, Np, alpha_range, n_harmonics=8, n_samples=8192,
                            n_samples_alpha0=None, seed=0, conjugate=True,
                            a=None, g=None, alpha0=None):
    """Estimate the complete alpha profile assuming a single harmonic comb.

    1. estimate alpha0 parametrically from a subsample (unless supplied),
    2. evaluate the profile at {m*alpha0 : m = -n_harmonics..n_harmonics}.

    `alpha_range` = (lo, hi) plausible fundamental range for the alpha0
    search (ignored if `alpha0` is given). The alpha0 scan dominates the
    cost and scales with (hi - lo) * N * n_samples_alpha0, so:
      * a TIGHTER `alpha_range` (a better baud-rate prior) is the biggest
        speedup -- narrowing from [0.3,3.5]*alpha0 to [0.9,1.1]*alpha0 cut
        the runtime ~80x with no accuracy loss in testing;
      * `n_samples_alpha0` (defaults to n_samples) can be smaller than the
        profile-evaluation sample count, since alpha0 estimation is robust
        and needs fewer samples than accurate magnitude evaluation.
    """
    x = np.asarray(x); N = x.size
    if a is None or g is None:
        a_d, g_d = _default_windows(N, Np)
        a = a_d if a is None else a
        g = g_d if g is None else g
    if n_samples_alpha0 is None:
        n_samples_alpha0 = n_samples

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(N - 1, min(n_samples, N - 1), replace=False))

    if alpha0 is None:
        idx_a0 = (idx if n_samples_alpha0 >= idx.size else
                  np.sort(rng.choice(idx, n_samples_alpha0, replace=False)))
        alpha0, _, _, _ = estimate_alpha0(x, alpha_range, idx=idx_a0, conjugate=conjugate)

    XT, _ = _channelizer_rows(x, Np, a, idx)
    Xg = XT * np.conj(x[idx])[:, None] * g[idx][:, None]

    m = np.arange(-n_harmonics, n_harmonics + 1)
    alphas = m * alpha0
    prof, peak_k = profile_at(Xg, idx, N, Np, alphas)
    elapsed = time.perf_counter() - t0
    return AlphaProfileResult(alphas, prof, peak_k, alpha0, idx.size, elapsed)


# ---------------------------------------------------------------------------
# self-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.getcwd())
    from bpsk_compare import bpsk_signal, three_coprime_near, nearest_pow2

    kappa, Np = 50, 32
    N = nearest_pow2(int(np.prod(three_coprime_near(kappa))))
    x, dr = bpsk_signal(N, 0.25, 31, 10.0, seed=0)
    print(f"DSSS-BPSK, N={N}, true alpha0={dr:.8f}")

    # dense reference, sampled at the true harmonics
    S, f, alpha = dense_ssca(x, Np)
    mag = np.abs(S); al = alpha.ravel(); mg = mag.ravel()
    def dref(a):
        band = np.abs(al - a) < 0.3 * dr
        return mg[band].max() if band.any() else 0.0

    res = harmonic_alpha_profile(x, Np, (dr * 0.5, dr * 1.5), n_harmonics=8,
                                 n_samples=8192, n_samples_alpha0=4096, seed=0)
    print(f"estimated alpha0={res.alpha0:.8f} (err {abs(res.alpha0-dr)/dr:.2%}), "
          f"used {res.n_samples_used} samples ({100*res.n_samples_used/N:.1f}% of N), "
          f"{res.elapsed*1e3:.0f} ms\n")

    print(f"{'m':>3}{'alpha':>11}{'dense P':>12}{'harmonic P':>12}{'rel err':>10}")
    for mm, aa, pp, pk in zip(range(-8, 9), res.alphas, res.profile, res.peak_channels):
        d = dref(mm * dr)
        err = abs(pp - d) / d if d > 0 else float("nan")
        print(f"{mm:>+3}{aa:>11.5f}{d:>12.1f}{pp:>12.1f}{err:>9.1%}")

    # overall accuracy on the populated comb
    errs = [abs(pp - dref(mm*dr))/dref(mm*dr)
            for mm, pp in zip(range(-8, 9), res.profile) if dref(mm*dr) > 0.05*mg.max()]
    print(f"\nmean rel err on populated harmonics: {np.mean(errs):.1%}")
