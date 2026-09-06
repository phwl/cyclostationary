"""
sFFT 1.0 -- Sparse Fast Fourier Transform.

Implementation of the algorithm in

    H. Hassanieh, P. Indyk, D. Katabi, E. Price,
    "Simple and Practical Algorithm for Sparse Fourier Transform", SODA 2012.

Given a length-n signal whose spectrum is dominated by k coefficients, this
recovers those coefficients while reading only O(loops * W) input samples,
W ~ 12*B << n.  Requires only NumPy.

DFT convention matches numpy.fft:   X[m] = sum_j x[j] exp(-2j*pi*j*m/n)

------------------------------------------------------------------------------
One inner loop
------------------------------------------------------------------------------
1. Random spectrum permutation.  For sigma invertible mod n and a shift tau,
   let y[i] = x[(sigma*i + tau) mod n].  Then, with w = exp(-2j*pi/n),

       Y[sigma*m] = w^(tau*m) * X[m]

   so frequency m moves to sigma*m and picks up a known phase.  Heavy
   coefficients, whatever their original layout, end up roughly uniformly
   spread -- which is what makes the hashing below behave.

2. Filtering and bucketization.  Multiply y by a flat window filter h whose
   support is W << n, then alias those W samples down to B buckets.  A
   length-B FFT gives

       Z[b] = U[b*n/B],   U = fft(h*y),   U[m] = (1/n) sum_a Gf[a] Y[m-a]

   Because |Gf| ~ 1 over one bucket width and ~0 beyond, bucket b is
   essentially the sum of the n/B permuted frequencies nearest b*n/B.  This is
   a hash of the spectrum into B buckets, computed in O(W + B log B) time.

3. Location.  The heaviest buckets are the ones likely to contain heavy
   coefficients, so every frequency hashing into one of the B_thresh heaviest
   buckets gets a vote.  Frequencies voted for in at least `loop_threshold` of
   the location loops (each with fresh sigma, tau) become candidates: a truly
   heavy coefficient lands in a heavy bucket every time, while an innocent
   bystander only does so when it happens to share a bucket.

4. Estimation.  For a candidate m, step 2 is inverted in every loop:

       X[m] ~ w^(tau*m) * Z[h(m)] / Gf[o(m)]

   where h(m) is m's bucket and o(m) its offset within the filter's response.
   Loops where m collided with another heavy coefficient give wild answers, so
   the estimates are combined with a coordinatewise median (real and imaginary
   parts separately) rather than a mean.

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
* B must divide n; powers of two for both is the easy case.
* Cost per loop is O(W + B log B + B_thresh*n/B) with W ~ 9*B, so the default
  B ~ sqrt(n*k/5) balances the filtering and candidate-enumeration terms.
  Shrinking B below that trades runtime for fewer samples read -- at n = 2^20,
  k = 50 the default reads ~0.4n samples in ~40 ms, while B = 512 with
  tolerance=1e-4 reads 0.06n samples but takes ~200 ms.  Both recover
  everything; pick the end of the curve your application cares about.
  (The paper's O(sqrt(n*k/log n)) differs by the log factor it gets from
  bounding the number of loops rather than fixing it, as here and in the
  authors' reference implementation.)
* Choosing B matters more than any other knob.  A bucket sums n/B
  frequencies, so the noise/tail energy folded into each one is about
  ||tail||_2 / sqrt(B); a coefficient is reliably located only when it stands
  above that.  The default B is tuned for near-exactly-sparse input.  For a
  signal with a substantial noise floor, raise B until
  ``||x - x_k||_2 * sqrt(n/B) << |X_m|`` -- extra loops do not substitute for
  this, since every loop faces the same folded noise.

* Building the filter costs one length-n FFT.  It depends only on (n, B), so
  pass the result back in via `filt=` when transforming many signals.
* This is not a speed contest with numpy.fft: FFTW-class C code beats a
  NumPy-level sparse implementation until n is enormous.  What is genuinely
  sublinear here is the number of input samples read, which is the property
  that matters when samples are expensive (spectrum sensing, GPS acquisition,
  MRI, wideband ADCs).
"""

from __future__ import annotations

from math import gcd
from typing import NamedTuple

import numpy as np

__all__ = ["Filter", "flat_filter", "sfft1", "sfft1_dense"]


# ---------------------------------------------------------------------------
# Windows and the flat filter
# ---------------------------------------------------------------------------
def _dolph_chebyshev(M: int, at_db: float) -> np.ndarray:
    """Length-M (odd) Dolph-Chebyshev window with `at_db` dB sidelobes."""
    if M % 2 == 0:
        raise ValueError("odd length required")
    order = M - 1
    beta = np.cosh(np.arccosh(10.0 ** (abs(at_db) / 20.0)) / order)
    x = beta * np.cos(np.pi * np.arange(M) / M)
    # Chebyshev polynomial T_order over the whole real line
    p = np.empty(M)
    hi, lo, mid = x > 1, x < -1, np.abs(x) <= 1
    p[hi] = np.cosh(order * np.arccosh(x[hi]))
    p[lo] = np.cosh(order * np.arccosh(-x[lo]))       # M odd => sign is +1
    p[mid] = np.cos(order * np.arccos(x[mid]))
    w = np.real(np.fft.fft(p))[: (M + 1) // 2]
    w = np.concatenate((w[:0:-1], w))
    return w / w.max()


class Filter(NamedTuple):
    """A flat window filter, reusable across calls with the same (n, B)."""
    h: np.ndarray     # time domain, supported on indices 0..W-1
    Gf: np.ndarray    # length-n response, fft(pad(h, n))/n, max|Gf| == 1
    W: int
    n: int
    B: int


def flat_filter(n, B, tolerance=1e-6, lobe_scale=0.5, box_scale=1.6) -> Filter:
    """Build a flat window filter that hashes a length-n spectrum into B buckets.

    |Gf| is ~1 across a band of about ``(box_scale - lobe_scale)*n/B``
    frequencies around 0 -- wide enough to cover a full bucket -- and decays to
    ~`tolerance` outside about ``box_scale*n/B``.

    A Dolph-Chebyshev window has a narrow main lobe with sidelobes at
    `tolerance`; convolving its response with a box-car of width
    ``box_scale*n/B`` flattens the top.  Convolution by a box-car in frequency
    is multiplication by a Dirichlet kernel in time, so it is applied directly
    to the W window samples and the support does not grow.
    """
    if n % B:
        raise ValueError(f"B={B} must divide n={n}")

    W = int(np.ceil(np.arccosh(1.0 / tolerance) * B / (np.pi * lobe_scale))) | 1
    W = min(W, n - 1 + (n & 1))
    half = W // 2
    win = _dolph_chebyshev(W, -20.0 * np.log10(tolerance))

    # Dirichlet kernel of the width-bw box-car, centred on the window
    bw = max(1, int(round(box_scale * n / B)))
    th = np.pi * np.arange(-half, half + 1) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.exp(2j * th * ((bw - 1) / 2.0 - bw // 2)) * np.sin(bw * th) / np.sin(th)
    d[half] = bw
    h = win * d

    padded = np.zeros(n, dtype=complex)
    padded[:W] = h
    Gf = np.fft.fft(padded)
    s = np.abs(Gf).max()
    return Filter(h * (n / s), Gf / s, W, n, B)


# ---------------------------------------------------------------------------
# sFFT 1.0
# ---------------------------------------------------------------------------
def sfft1(
    x,
    k,
    B=None,
    loc_loops=4,
    est_loops=16,
    loop_threshold=None,
    B_thresh=None,
    tolerance=1e-6,
    box_scale=1.6,
    gf_floor=0.1,
    filt=None,
    rng=None,
):
    """Estimate the k largest Fourier coefficients of `x`.

    Parameters
    ----------
    x : array_like, length n.
    k : target sparsity.
    B : number of buckets, must divide n.  Default ~sqrt(n*k/6), rounded down
        to a power of two dividing n.
    loc_loops, est_loops : number of location loops and additional estimation
        loops.  Location loops also contribute bucket values to estimation, so
        each candidate gets loc_loops + est_loops estimates.  More location
        loops sharpen which frequencies are proposed; more estimation loops
        sharpen the values.
    loop_threshold : votes needed to become a candidate, out of loc_loops
        (default loc_loops//2 + 1).  Raise it to cut false candidates, lower
        it if heavy coefficients are being missed.
    B_thresh : buckets kept per location loop (default 2*k, capped at B).
    tolerance, box_scale : filter sidelobe level and flat-top width in units
        of n/B.  Ignored if `filt` is given.
    gf_floor : estimates whose filter weight falls below this are dropped
        instead of divided through; the median is taken over the rest.
    filt : a `Filter` from `flat_filter` to reuse (skips filter construction).
    rng : seed or numpy Generator, for reproducibility.

    Returns
    -------
    freqs : int array of recovered frequency indices, descending by magnitude.
    coeffs : complex array of estimates, scaled like numpy.fft.fft(x).
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    x = x.astype(complex, copy=False)
    n = x.size
    rng = np.random.default_rng(rng)
    k = int(min(k, n))

    if filt is not None:
        if filt.n != n:
            raise ValueError(f"filter built for n={filt.n}, got n={n}")
        B = filt.B
    if B is None:
        B = 1 << max(1, int(round(0.5 * np.log2(max(n * k / 5.0, 4)))))
        B = min(B, n)
        while n % B:
            B //= 2
    if n % B:
        raise ValueError(f"B={B} must divide n={n}")
    if filt is None:
        filt = flat_filter(n, B, tolerance=tolerance, box_scale=box_scale)

    B_thresh = int(min(B_thresh if B_thresh is not None else 2 * k, B))
    if loop_threshold is None:
        loop_threshold = loc_loops // 2 + 1
    loop_threshold = int(min(max(loop_threshold, 1), loc_loops))

    h, Gf, W = filt.h, filt.Gf, filt.W
    supp = np.arange(W)
    supp_bucket = supp % B                      # aliasing map, precomputed
    bucket_width = n // B
    step = np.arange(bucket_width)

    cand_chunks, loops = [], []

    for loop in range(loc_loops + est_loops):
        sigma = int(rng.integers(0, n))
        while gcd(sigma, n) != 1:                # invertible mod n
            sigma = int(rng.integers(0, n))
        tau = int(rng.integers(0, n))

        # --- permute, filter, alias into B buckets, length-B FFT ----------
        u = h * x[(sigma * supp + tau) % n]
        z = np.zeros(B, dtype=complex)
        np.add.at(z, supp_bucket, u)
        Z = np.fft.fft(z)
        loops.append((sigma, tau, Z))

        # --- location: propose everything in the heaviest buckets ---------
        # Preimages of distinct buckets are disjoint, so a frequency collects
        # at most one vote per loop and the votes can be tallied at the end.
        if loop < loc_loops:
            sigma_inv = pow(sigma, -1, n)
            top = np.argpartition(-np.abs(Z), B_thresh - 1)[:B_thresh]
            low = np.ceil((top - 0.5) * bucket_width).astype(np.int64)
            perm = (low[:, None] + step[None, :]).ravel()
            cand_chunks.append((sigma_inv * perm) % n)

    freqs, votes = np.unique(np.concatenate(cand_chunks), return_counts=True)
    freqs = freqs[votes >= loop_threshold]
    if freqs.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=complex)

    # --- estimation: invert the hash in every loop, then take medians -----
    ests = np.empty((len(loops), freqs.size), dtype=complex)
    for j, (sigma, tau, Z) in enumerate(loops):
        perm = (sigma * freqs) % n
        b = np.rint(perm / bucket_width).astype(np.int64) % B
        weight = Gf[(b * bucket_width - perm) % n]
        e = Z[b] * np.exp(-2j * np.pi * tau * freqs / n) / weight
        ests[j] = np.where(np.abs(weight) >= gf_floor, e, np.nan)

    with np.errstate(invalid="ignore"):
        coeffs = np.nanmedian(ests.real, axis=0) + 1j * np.nanmedian(ests.imag, axis=0)
    coeffs = np.nan_to_num(coeffs)

    order = np.argsort(-np.abs(coeffs))[:k]
    order = order[np.abs(coeffs[order]) > 0]
    return freqs[order], coeffs[order]


def sfft1_dense(x, k, **kw) -> np.ndarray:
    """`sfft1` packed into a length-n, mostly-zero spectrum."""
    freqs, coeffs = sfft1(x, k, **kw)
    out = np.zeros(len(x), dtype=complex)
    out[freqs] = coeffs
    return out


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)

    def sparse_signal(n, k, snr_db=None):
        X = np.zeros(n, dtype=complex)
        freqs = rng.choice(n, k, replace=False)
        v = rng.standard_normal(k) + 1j * rng.standard_normal(k)
        X[freqs] = v * (1 + rng.random(k)) / np.abs(v)
        x = np.fft.ifft(X)
        if snr_db is not None:
            s = np.sqrt(np.mean(np.abs(x) ** 2) / (2 * 10 ** (snr_db / 10)))
            x = x + s * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        return x, X, freqs

    print("exactly sparse signals")
    print(f"{'n':>10} {'k':>5} {'found':>7} {'coeff err':>11} "
          f"{'samples':>10} {'frac of n':>10} {'sfft':>9} {'np.fft':>9}")
    for n, k in [(1 << 14, 20), (1 << 16, 50), (1 << 18, 50), (1 << 20, 100)]:
        x, X, freqs = sparse_signal(n, k)
        t = time.perf_counter()
        f_hat, c_hat = sfft1(x, k, rng=1)
        t_s = time.perf_counter() - t
        t = time.perf_counter()
        np.fft.fft(x)
        t_f = time.perf_counter() - t
        found = len(set(f_hat.tolist()) & set(freqs.tolist()))
        err = np.max(np.abs(c_hat - X[f_hat])) / np.max(np.abs(X))
        # input samples read = (loc_loops + est_loops) * filter support
        B = min(1 << max(1, int(round(0.5 * np.log2(n * k / 5.0)))), n)
        used = 20 * flat_filter(n, B).W
        print(f"{n:>10} {k:>5} {found:>4}/{k:<2} {err:>11.2e} {used:>10} "
              f"{used / n:>9.2f}x {t_s * 1e3:>8.0f}ms {t_f * 1e3:>8.0f}ms")

    print("\nsame signal, filter tuned for sample frugality (n = 2^20, k = 50)")
    n, k = 1 << 20, 50
    x, X, freqs = sparse_signal(n, k)
    frugal = flat_filter(n, 512, tolerance=1e-4)
    t = time.perf_counter()
    f_hat, c_hat = sfft1(x, k, filt=frugal, rng=1)
    t_s = time.perf_counter() - t
    found = len(set(f_hat.tolist()) & set(freqs.tolist()))
    print(f"  {found}/{k} located reading {20 * frugal.W} samples "
          f"({20 * frugal.W / n:.3f}n) in {t_s * 1e3:.0f} ms")

    print("\ndense noise floor: why B is the knob that matters (n = 2^18, k = 30)")
    n, k = 1 << 18, 30
    X = np.zeros(n, dtype=complex)
    freqs = rng.choice(n, k, replace=False)
    X[freqs] = (3 + rng.random(k)) * np.exp(2j * np.pi * rng.random(k))
    x = np.fft.ifft(X) + 2e-4 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    tail = np.linalg.norm(np.delete(np.fft.fft(x), freqs))
    print(f"  coefficients are 3.0-4.0 in magnitude, ||tail||_2 = {tail:.0f}")
    for B in (1 << 10, 1 << 12, 1 << 14):
        f_hat, _ = sfft1(x, k, filt=flat_filter(n, B), rng=0)
        found = len(set(f_hat.tolist()) & set(freqs.tolist()))
        print(f"  B = {B:>6}: noise per bucket ~{tail / np.sqrt(B):.2f}  ->  "
              f"{found:>2}/{k} located")

    print("\nnoisy signals (n = 2^18, k = 40)")
    for snr in (30, 20, 10, 0):
        x, X, freqs = sparse_signal(1 << 18, 40, snr_db=snr)
        f_hat, c_hat = sfft1(x, 40, rng=2)
        found = len(set(f_hat.tolist()) & set(freqs.tolist()))
        rel = np.linalg.norm(c_hat - X[f_hat]) / np.linalg.norm(X[freqs])
        print(f"  SNR {snr:>3} dB: {found:>2}/40 located, relative error {rel:.4f}")

    print("\nreusing one filter over many signals (n = 2^18, k = 30)")
    n, k = 1 << 18, 30
    f = flat_filter(n, 1 << 11)
    t = time.perf_counter()
    missed = 0
    for s in range(20):
        x, X, freqs = sparse_signal(n, k)
        f_hat, _ = sfft1(x, k, filt=f, rng=s)
        missed += k - len(set(f_hat.tolist()) & set(freqs.tolist()))
    print(f"  20 transforms, {missed} of {20 * k} coefficients missed, "
          f"{(time.perf_counter() - t) / 20 * 1e3:.0f} ms each")
