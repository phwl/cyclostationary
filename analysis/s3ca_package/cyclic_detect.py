"""
cyclic_detect.py -- parametric detection and estimation of cyclic features,
as an alternative to sparse-FFT *reconstruction* of the SCD.

------------------------------------------------------------------------------
Why this is a different problem (and why the reconstruction metrics were
unfair to it)
------------------------------------------------------------------------------
Every other method in this project (sfft1, decimated, rffast, harmonic,
debias) tries to RECONSTRUCT the SCD surface -- recover a set of
(frequency, cycle-frequency, value) coefficients -- and is then scored by
how many of the dense SCD's top-kappa peaks it recovers ("hit rate"). For
an APPROXIMATELY sparse signal like DSSS-BPSK that surface has a heavy tail
(top-50 coefficients hold only ~84% of a channel's energy), which is
exactly what defeats the exactly-sparse reconstructors.

But for detection and classification you usually do not NEED the surface.
You need a few numbers: IS there a cyclic feature, and at what cycle
frequency (baud rate) alpha0? That is a PARAMETRIC estimation problem, and
it is indifferent to the broadband tail because it never tries to represent
it. This module estimates the cyclic autocorrelation directly at candidate
cycle frequencies:

    R_x^alpha(tau) = (1/M) sum_{t in idx} x(t+tau) x*(t) exp(-i 2 pi alpha t)

A cyclostationary signal has |R_x^alpha(tau)| well above the noise floor
only at alpha = m*alpha0 (and, for the *conjugate* cyclic autocorrelation
x(t+tau)x(t), at conjugate cycle frequencies). Everywhere else it is ~0 --
including all across the heavy tail that broke reconstruction. On the same
BPSK signal where reconstruction hit rate was 5-73%, the cyclic
autocorrelation at the true harmonics is HUNDREDS to THOUSANDS of times the
off-grid floor (see the accompanying notebook / self-test).

This is the classical cyclostationary-detection approach (Gardner's cyclic
autocorrelation / spectral coherence), here (a) driven from a *subsampled*
set of time positions to keep the cost sublinear, and (b) combined across
several lags into a single detection statistic. It connects to the subspace
methods (MUSIC/ESPRIT) mentioned as "option 5": those operate on the same
cyclic-autocorrelation quantities; this module uses a simpler coherent
lag-combining statistic, which is enough to show the qualitative point --
parametric detection sidesteps the sparsity problem entirely.

------------------------------------------------------------------------------
What it does and does not give you
------------------------------------------------------------------------------
* GIVES: presence/absence of a cyclic feature (a detection statistic with a
  controllable false-alarm rate), and an estimate of the fundamental cycle
  frequency alpha0, from sublinear samples, robust to the broadband tail.
* DOES NOT GIVE: the full SCD surface. If you actually need S_X^alpha(f)
  over the whole plane (not just the cyclic feature locations), this is the
  wrong tool -- use a reconstruction method. Detection/classification is
  its purpose, and on that task the "hit rate" metric simply does not
  apply.

------------------------------------------------------------------------------
Honest limitations
------------------------------------------------------------------------------
* A caller-supplied search grid / range for alpha0 is used (as in
  harmonic.py). The scan is O(n_alpha * n_lags * n_samples); the coarse-to-
  fine option keeps that manageable.
* Lag selection matters: lags near the symbol/chip period carry the
  strongest cyclic feature. Defaults probe a small spread of small lags;
  for a known modulation you would tune these.
* The detection threshold here is set from the empirical off-grid
  statistic distribution (a CFAR-style floor estimate), not a closed-form
  analytic threshold -- adequate to demonstrate ROC behaviour, but a
  deployment would derive the threshold from the known statistic
  distribution under H0.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

__all__ = ["cyclic_autocorr", "detection_statistic", "scan_alpha",
           "estimate_alpha0", "CyclicDetector", "roc_points"]


# ---------------------------------------------------------------------------
# core cyclic autocorrelation on a subsampled index set
# ---------------------------------------------------------------------------
def cyclic_autocorr(x, alpha, tau, idx=None, conjugate=False):
    """Estimate R_x^alpha(tau) from samples at positions `idx` (all samples
    if idx is None).

    conjugate=False : x(t+tau) * conj(x(t))   (ordinary cyclic autocorr)
    conjugate=True  : x(t+tau) * x(t)         (conjugate cyclic autocorr;
                      for BPSK the conjugate features are strong and this is
                      often the more informative one)
    """
    x = np.asarray(x)
    N = x.size
    if idx is None:
        idx = np.arange(N - abs(tau))
    else:
        idx = idx[idx + tau < N]
    if conjugate:
        prod = x[idx + tau] * x[idx]
    else:
        prod = x[idx + tau] * np.conj(x[idx])
    return np.mean(prod * np.exp(-2j * np.pi * alpha * idx))


def detection_statistic(x, alpha, taus, idx=None, conjugate=False):
    """Coherent-across-lags detection statistic at a single candidate alpha:
    sqrt(sum_tau |R_x^alpha(tau)|^2). Combining several lags averages down
    the noise floor while the true cyclic feature adds up across lags."""
    return np.sqrt(np.sum([np.abs(cyclic_autocorr(x, alpha, t, idx, conjugate)) ** 2
                           for t in taus]))


# ---------------------------------------------------------------------------
# scanning / estimation
# ---------------------------------------------------------------------------
def scan_alpha(x, alphas, taus, idx=None, conjugate=False, max_elems=50_000_000):
    """Detection statistic over a grid of candidate cycle frequencies.
    Vectorized across alphas: for each lag it forms the lag product once and
    evaluates all cycle frequencies via a single matrix product, rather than
    looping alpha-by-alpha (which was the dominant cost of the scan).

    The (n_alpha x n_samples) phase matrix can be large, so alphas are
    processed in chunks capped at ~max_elems complex entries to bound peak
    memory (the full-sample, dense-grid case would otherwise need many GB).
    """
    x = np.asarray(x)
    N = x.size
    alphas = np.asarray(alphas)
    acc = np.zeros(alphas.size)
    for tau in taus:
        if idx is None:
            ii = np.arange(N - abs(tau))
        else:
            ii = idx[idx + tau < N]
        prod = x[ii + tau] * (x[ii] if conjugate else np.conj(x[ii]))
        chunk = max(1, int(max_elems // max(ii.size, 1)))
        for s in range(0, alphas.size, chunk):
            a = alphas[s:s + chunk]
            phase = np.exp(-2j * np.pi * np.outer(a, ii))
            R = phase @ prod / ii.size
            acc[s:s + chunk] += np.abs(R) ** 2
    return np.sqrt(acc)


def estimate_alpha0(x, alpha_range, taus=(1, 2, 3, 4, 5, 6, 7, 8), idx=None,
                     conjugate=False, coarse=None, refine=True, refine_factor=50,
                     harmonic_aware=True):
    """Estimate the fundamental cycle frequency alpha0 in `alpha_range =
    (lo, hi)`.

    The cyclic-autocorrelation peak at a true cycle frequency is as sharp as
    a full-resolution DFT bin (width ~1/N in alpha). A too-coarse grid lets
    the sharp peak fall between samples so a nearby harmonic that happens to
    land on a grid point wins instead (a real bug this hit: a 400-point grid
    locked onto 2*alpha0). Two safeguards:

    * `coarse=None` sizes the grid so the step is ~1/(2N) -- fine enough that
      no sharp cyclic peak hides between samples.
    * `harmonic_aware=True` adds a harmonic-sum disambiguation: rather than
      taking the single tallest scan peak (which can be a strong harmonic
      m*alpha0 rather than the fundamental), it re-scores the top peaks by
      how much cyclic energy sits at that candidate AND its first few
      sub-multiples, preferring the smallest alpha that explains the comb.
      This is the cyclic-domain analogue of the subharmonic tie-break in
      harmonic.py.

    Returns (alpha0_hat, peak_stat, alphas_scanned, stats_scanned).
    """
    lo, hi = alpha_range
    N = np.asarray(x).size
    if coarse is None:
        coarse = max(64, int(np.ceil((hi - lo) * N * 2)))
    alphas = np.linspace(lo, hi, coarse)
    stats = scan_alpha(x, alphas, taus, idx, conjugate)

    if harmonic_aware:
        # consider the strongest few scan peaks as fundamental candidates,
        # and for each, sum the statistic at it and its first few harmonics
        # (m=1..H). A true fundamental lights up its whole comb; a harmonic
        # of the true fundamental cannot (its own "harmonics" would fall
        # outside / between real lines). Prefer the candidate with the
        # largest comb sum, breaking ties toward smaller alpha.
        n_peaks = min(8, stats.size)
        cand_i = np.argsort(-stats)[:n_peaks]
        H = 4
        best, best_score = None, -np.inf
        for ci in cand_i:
            a = alphas[ci]
            comb = np.array([a * m for m in range(1, H + 1) if a * m <= hi])
            if comb.size == 0:
                continue
            comb_stat = scan_alpha(x, comb, taus, idx, conjugate).sum()
            # slight preference for smaller alpha to resolve fundamental vs
            # its harmonics when comb sums are close
            score = comb_stat - 1e-9 * a
            if score > best_score:
                best_score, best = score, a
        peakloc = best if best is not None else alphas[int(np.argmax(stats))]
        i = int(np.argmin(np.abs(alphas - peakloc)))
    else:
        i = int(np.argmax(stats))
    best = alphas[i]

    if refine and coarse > 1:
        step = alphas[1] - alphas[0]
        fine = np.linspace(best - step, best + step, refine_factor)
        fine = fine[(fine > 0)]
        fstats = scan_alpha(x, fine, taus, idx, conjugate)
        j = int(np.argmax(fstats))
        best, peak = fine[j], fstats[j]
        return best, peak, alphas, stats
    return best, stats[i], alphas, stats


# ---------------------------------------------------------------------------
# detector object with a CFAR-style threshold
# ---------------------------------------------------------------------------
class DetectionResult(NamedTuple):
    detected: bool
    alpha0_hat: float
    peak_stat: float
    threshold: float
    noise_floor: float
    n_samples_used: int
    elapsed: float


class CyclicDetector:
    """Parametric cyclic-feature detector.

    Parameters
    ----------
    N : signal length it will be applied to.
    alpha_range : (lo, hi) plausible fundamental cycle-frequency range.
    n_samples : how many time positions to subsample (sublinear cost). None
        uses all N.
    taus : lags to combine in the statistic.
    conjugate : use the conjugate cyclic autocorrelation (often stronger for
        real-valued digital modulations like BPSK).
    pfa : target false-alarm probability, used to set the threshold from the
        empirical off-grid (H0-like) statistic distribution.
    seed : subsampling RNG seed.
    """

    def __init__(self, N, alpha_range, n_samples=4096, taus=(1, 2, 3, 4, 5, 6, 7, 8),
                 conjugate=True, pfa=1e-3, seed=0):
        self.N = N
        self.alpha_range = alpha_range
        self.taus = taus
        self.conjugate = conjugate
        self.pfa = pfa
        rng = np.random.default_rng(seed)
        max_tau = max(taus)
        self.idx = (np.sort(rng.choice(N - max_tau, min(n_samples, N - max_tau),
                                       replace=False))
                    if n_samples else None)
        self.n_samples = self.idx.size if self.idx is not None else N

    def _threshold_from_scan(self, stats):
        """CFAR-style threshold from the coarse-scan statistics themselves.
        The overwhelming majority of grid points are off-harmonic and so
        behave like H0 (no cyclic feature); a robust center+spread of the
        whole scan estimates the H0 distribution without a separate sampling
        pass (and without the earlier bug where hand-picked 'off-grid'
        alphas occasionally landed on true harmonics and inflated the
        floor, wrecking the false-alarm rate). Threshold = median + k*MAD
        with k chosen for the target Pfa under an approximately-Rayleigh
        statistic."""
        med = np.median(stats)
        mad = np.median(np.abs(stats - med)) + 1e-30
        sigma = 1.4826 * mad
        # k robust-sigmas for the target Pfa. The detection statistic is the
        # MAX over the scan grid, so its H0 tail is heavier than a single
        # look -- but the grid points are strongly correlated (neighbouring
        # alphas overlap), so the naive Bonferroni count over all n_grid
        # points is far too conservative (empirically it pushed k to ~5.2
        # and killed sensitivity, while the measured H0 peak z-score sits
        # near ~4.1 for Pfa~1e-3). Use a base multiplier calibrated to the
        # empirical H0 max-z distribution, with only a weak (log-log) grid
        # term so very large grids still get a small correction.
        n_grid = max(stats.size, 2)
        k = 3.6 + 0.35 * np.log(np.log(n_grid) + 1.0) + 0.4 * np.log10(1e-3 / self.pfa)
        thr = med + k * sigma
        return thr, med

    def detect(self, x) -> DetectionResult:
        t0 = time.perf_counter()
        alpha0_hat, peak, _, stats = estimate_alpha0(
            x, self.alpha_range, self.taus, self.idx, self.conjugate)
        thr, floor = self._threshold_from_scan(stats)
        elapsed = time.perf_counter() - t0
        return DetectionResult(
            detected=bool(peak > thr), alpha0_hat=alpha0_hat, peak_stat=peak,
            threshold=thr, noise_floor=floor, n_samples_used=self.n_samples,
            elapsed=elapsed)


# ---------------------------------------------------------------------------
# ROC helper
# ---------------------------------------------------------------------------
def roc_points(signal_fn, noise_fn, detector: CyclicDetector, n_trials=100):
    """Empirical (Pfa, Pd) at the detector's current threshold, plus the raw
    statistic distributions, by running it over `n_trials` signal-present
    and signal-absent realizations.

    signal_fn(i) -> a signal-present realization (with cyclic feature)
    noise_fn(i)  -> a signal-absent realization (noise only)
    """
    peaks_H1, peaks_H0, thr_acc = [], [], []
    for i in range(n_trials):
        xs = signal_fn(i)
        r = detector.detect(xs)
        peaks_H1.append(r.peak_stat); thr_acc.append(r.threshold)
        xn = noise_fn(i)
        rn = detector.detect(xn)
        peaks_H0.append(rn.peak_stat)
    peaks_H1 = np.array(peaks_H1); peaks_H0 = np.array(peaks_H0)
    # sweep threshold to trace a full ROC
    all_stats = np.concatenate([peaks_H0, peaks_H1])
    thrs = np.linspace(all_stats.min(), all_stats.max(), 200)
    pd = np.array([(peaks_H1 > t).mean() for t in thrs])
    pfa = np.array([(peaks_H0 > t).mean() for t in thrs])
    return {"peaks_H1": peaks_H1, "peaks_H0": peaks_H0,
            "pd": pd, "pfa": pfa, "thresholds": thrs}


# ---------------------------------------------------------------------------
# self-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.getcwd())
    from bpsk_compare import bpsk_signal

    N = 131072
    x, dr = bpsk_signal(N, 0.25, 31, 10.0, seed=0)
    print(f"DSSS-BPSK, N={N}, true alpha0 = {dr:.8f}")
    print()

    # 1) estimation accuracy vs subsample count (subsampled = the intended,
    #    sublinear-cost regime; the scan cost is ~ n_alpha * n_samples, so
    #    the small-sample rows are both the fast ones AND the intended ones)
    print("alpha0 estimation from subsampled data:")
    rng = np.random.default_rng(0)
    for nsub in (4096, 1024, 256):
        idx = np.sort(rng.choice(N - 8, nsub, replace=False))
        a_hat, peak, _, _ = estimate_alpha0(x, (dr * 0.3, dr * 3.5), idx=idx,
                                            conjugate=True)
        err = abs(a_hat - dr) / dr
        print(f"  {nsub:>6} samples ({100*nsub/N:5.1f}% of N): "
              f"alpha0_hat = {a_hat:.8f}  rel err = {err:.2%}")
    print()

    # 2) detection at low SNR (smaller working N keeps the dense-grid scan
    #    fast; the alpha0-estimation accuracy above already used full N)
    print("detection vs SNR (Nd=16384, 2048 samples = 12.5%, Pfa target 1e-3):")
    Nd = 16384
    _, drd = bpsk_signal(Nd, 0.25, 31, 10.0, seed=0)
    for snr in (10, 7, 5, 3, 0, -5):
        det = CyclicDetector(Nd, (drd * 0.3, drd * 3.5), n_samples=2048,
                             conjugate=True, pfa=1e-3, seed=1)
        xs, _ = bpsk_signal(Nd, 0.25, 31, float(snr), seed=snr + 100)
        r = det.detect(xs)
        margin = r.peak_stat / r.threshold
        print(f"  SNR {snr:>4} dB: detected={r.detected!s:>5}  "
              f"peak/threshold = {margin:5.2f}  alpha0_hat = {r.alpha0_hat:.6f}"
              f"  ({r.elapsed*1e3:.0f} ms)")
    print()

    # 3) false-alarm check on noise-only inputs
    print("false-alarm check on noise-only inputs (target Pfa 1e-3):")
    det = CyclicDetector(Nd, (drd * 0.3, drd * 3.5), n_samples=2048,
                         conjugate=True, pfa=1e-3, seed=1)
    fa = 0
    ntest = 50
    for i in range(ntest):
        rng2 = np.random.default_rng(1000 + i)
        noise = (rng2.standard_normal(Nd) + 1j * rng2.standard_normal(Nd)) / np.sqrt(2)
        if det.detect(noise).detected:
            fa += 1
    print(f"  false alarms: {fa}/{ntest}")
