"""
sFFT 1.0 -- Sparse Fast Fourier Transform, optimized.

Implementation of the algorithm in

    H. Hassanieh, P. Indyk, D. Katabi, E. Price,
    "Simple and Practical Algorithm for Sparse Fourier Transform", SODA 2012.

For a length-n signal dominated by k Fourier coefficients this recovers them in
sublinear time, reading only O(loops * W) input samples with W ~ 6*B << n.
NumPy only.  DFT convention matches numpy.fft:

    X[m] = sum_j x[j] exp(-2j*pi*j*m/n)

------------------------------------------------------------------------------
One inner loop
------------------------------------------------------------------------------
1. Random spectrum permutation.  For sigma invertible mod n and a shift tau,
   let y[i] = x[(sigma*i + tau) mod n].  Then, with w = exp(-2j*pi/n),

       Y[sigma*m] = w^(tau*m) * X[m]

   so frequency m moves to sigma*m and picks up a known phase.  Heavy
   coefficients, whatever their original layout, get spread roughly uniformly,
   which is what makes the hashing below behave.

2. Filtering and bucketization.  Multiply y by a flat window filter h of
   support W << n, alias those W samples down to B buckets, and take a
   length-B FFT:

       Z[b] = U[b*n/B],   U = fft(h*y),   U[m] = (1/n) sum_a Gf[a] Y[m-a]

   Since |Gf| ~ 1 across one bucket width and ~0 beyond, bucket b is
   essentially the sum of the n/B permuted frequencies nearest b*n/B.  This
   hashes the spectrum into B buckets in O(W + B log B) time.

3. Location.  Heavy buckets are the ones likely to hold heavy coefficients, so
   every frequency hashing into one of the B_thresh heaviest buckets gets a
   vote.  Frequencies voted for in at least `loop_threshold` of the location
   loops (each with fresh sigma, tau) become candidates: a genuinely heavy
   coefficient lands in a heavy bucket every time, an innocent bystander only
   when it happens to share one.

4. Estimation.  For a candidate m, step 2 is inverted in every loop,

       X[m] ~ w^(tau*m) * Z[h(m)] / Gf[o(m)]

   with h(m) the bucket and o(m) the offset within the filter response.  Loops
   where m collided with another heavy coefficient give wild answers, so the
   estimates are combined by a coordinatewise median (real and imaginary parts
   separately) rather than a mean.

------------------------------------------------------------------------------
Performance
------------------------------------------------------------------------------
Measured against single-threaded numpy.fft.fft (complex128), k = 50:

        n         np.fft      sfft1     speedup
        2^18       6.6 ms    13.2 ms      0.5x
        2^20      37.8 ms    24.2 ms      1.6x
        2^22     172.9 ms    67.8 ms      2.6x
        2^24     739.1 ms    53.3 ms     13.9x    (tolerance=1e-4, est_loops=8)

and the crossover in sparsity, at tolerance=1e-4 and est_loops=8:

        n = 2^22:  faster up to about k = 3000
        n = 2^24:  faster up to about k = 10000

Beyond those, the dense FFT wins and should be used.  Caveats worth knowing:
the baseline here is single-threaded, and a threaded FFT (scipy.fft with
workers, or FFTW) shifts the crossover left roughly in proportion to the core
count, since the loops below are not parallelized either; and O(n) work
remains in the vote table, so this is sublinear in *samples read* but not in
memory footprint.

Three things make it fast:

* No length-n transform anywhere.  The Dolph-Chebyshev window's frequency
  response is known in closed form, T_{W-1}(beta*cos(pi*a/n)), so the flat
  filter's response is a running sum of that -- evaluated only on the
  bucket-wide range of offsets the estimator actually asks for, in O(W + n/B)
  instead of O(n log n).  A single length-n FFT for the filter alone would
  cost as much as the dense transform it is trying to beat.
* The W filter taps are padded to a multiple of B, so folding into buckets is
  a reshape and a sum along an axis rather than a scatter-add (`np.add.at` is
  roughly 5x slower here), and all loops are batched into one gather and one
  batched FFT to amortize Python overhead.
* Votes are tallied with a plain fancy-indexed increment.  Bucket preimages
  are disjoint, so a frequency can be voted for at most once per loop and the
  increment needs no scatter-add semantics.

------------------------------------------------------------------------------
Tuning
------------------------------------------------------------------------------
* B, the number of buckets, is the knob that matters most.  A bucket sums n/B
  frequencies, so the noise/tail energy folded into each is about
  ||tail||_2 / sqrt(B), and a coefficient is located reliably only when it
  stands above that.  The default suits near-exactly-sparse input; for a
  signal with a real noise floor, raise B until
  ``||x - x_k||_2 * sqrt(n/B) << |X_m|``.  Extra loops do not substitute --
  every loop faces the same folded noise.  Runtime is flat within about a
  factor of two of the default, so raising B is usually cheap.
* `tolerance` sets the filter's sidelobe level and hence W ~ log(1/tolerance)*B
  and the achievable accuracy: 1e-6 gives ~5e-7 relative error, 1e-4 gives
  ~1e-5 for ~30% less work.
* `est_loops` can drop from 16 to 8 for exactly-sparse input.  Lowering
  `loc_loops` is a false economy: `loop_threshold` falls with it, more false
  candidates survive, and estimation gets slower than the loops saved.
* Filters depend only on (n, B), so pass one back via `filt=` when
  transforming many signals of the same size.
"""

from __future__ import annotations

from math import gcd
from typing import NamedTuple

import numpy as np

__all__ = ["Filter", "flat_filter", "sfft1", "sfft1_dense"]


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def _cheb_T(order, x):
    """Chebyshev polynomial T_order, valid over the whole real line."""
    out = np.empty(np.shape(x), dtype=float)
    hi, lo, mid = x > 1, x < -1, np.abs(x) <= 1
    out[hi] = np.cosh(order * np.arccosh(x[hi]))
    out[lo] = (-1.0) ** order * np.cosh(order * np.arccosh(-x[lo]))
    out[mid] = np.cos(order * np.arccos(x[mid]))
    return out


def _dolph_chebyshev(M, at_db):
    """Length-M (odd) Dolph-Chebyshev window with `at_db` dB sidelobes."""
    if M % 2 == 0:
        raise ValueError("odd length required")
    p = _cheb_T(M - 1, _cheb_beta(M, at_db) * np.cos(np.pi * np.arange(M) / M))
    w = np.real(np.fft.fft(p))[: (M + 1) // 2]
    return np.concatenate((w[:0:-1], w)) / w.max()


def _cheb_beta(M, at_db):
    return np.cosh(np.arccosh(10.0 ** (abs(at_db) / 20.0)) / (M - 1))


# ---------------------------------------------------------------------------
# Flat window filter
# ---------------------------------------------------------------------------
class Filter(NamedTuple):
    """A flat window filter; reusable across any signal with the same (n, B)."""
    h: np.ndarray      # time domain, length Wp (a multiple of B, zero padded)
    Gf: np.ndarray     # response at offsets -M2..M2, normalised to max 1
    M2: int
    W: int             # true support
    Wp: int            # padded support
    n: int
    B: int


def flat_filter(n, B, tolerance=1e-6, lobe_scale=0.5, box_scale=1.6, dtype=complex):
    """Build a flat window filter hashing a length-n spectrum into B buckets.

    |Gf| is ~1 across a band of about ``(box_scale - lobe_scale)*n/B``
    frequencies around 0 -- enough to cover a full bucket -- and decays to
    ~`tolerance` outside about ``box_scale*n/B``.

    A Dolph-Chebyshev window has a narrow main lobe with sidelobes at
    `tolerance`; convolving its response with a box-car of width
    ``box_scale*n/B`` flattens the top.  In the time domain that convolution is
    multiplication by a Dirichlet kernel, so it costs O(W) and does not grow
    the support.  In the frequency domain the window's response is the closed
    form ``T_{W-1}(beta*cos(pi*a/n))``, so the box-car convolution is a running
    sum of it -- and only the offsets the estimator can ask for, |a| <= n/2B,
    need computing.  Total cost O(W + n/B), with no length-n transform.
    """
    if n % B:
        raise ValueError(f"B={B} must divide n={n}")
    at_db = -20.0 * np.log10(tolerance)
    W = int(np.ceil(np.arccosh(1.0 / tolerance) * B / (np.pi * lobe_scale))) | 1
    W = min(W, n - 1 + (n & 1))
    half, order = W // 2, W - 1

    # --- time domain: window times the Dirichlet kernel of the box-car ----
    bw = max(1, int(round(box_scale * n / B)))
    c = bw // 2
    th = np.pi * np.arange(-half, half + 1) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.exp(2j * th * ((bw - 1) / 2.0 - c)) * np.sin(bw * th) / np.sin(th)
    d[half] = bw
    h = _dolph_chebyshev(W, at_db) * d

    # --- frequency domain: running sum of the closed-form response --------
    # DTFT(h)[a] = sum_{m=0..bw-1} G[a + c - m],  G[a] = T_order(beta cos(pi a/n))
    M2 = n // B // 2 + 2
    bins = np.arange(-M2 + c - bw + 1, M2 + c + 1)
    G = _cheb_T(order, _cheb_beta(W, at_db) * np.cos(np.pi * bins / n))
    cs = np.concatenate(([0.0], np.cumsum(G)))
    H = cs[bw:] - cs[:-bw]                                  # offsets -M2..M2
    Gf = H * np.exp(-2j * np.pi * np.arange(-M2, M2 + 1) * half / n)

    # The closed form is right up to one overall constant; pin it down against
    # the true DTFT at offset 0, which is just sum(h).
    kappa = h.sum() / H[M2]
    s = np.abs(Gf).max()

    Wp = -(-W // B) * B                    # pad so folding is a reshape-sum
    hp = np.zeros(Wp, dtype=dtype)
    hp[:W] = h * (n / (kappa * s))
    return Filter(hp, (Gf / s).astype(complex), M2, W, Wp, n, B)


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
    max_candidates=None,
    filt=None,
    rng=None,
    batch=8,
):
    """Estimate the k largest Fourier coefficients of `x`.

    Parameters
    ----------
    x : array_like, length n.
    k : target sparsity.
    B : number of buckets, must divide n.  Default ~sqrt(n*k/5) rounded to a
        power of two.  Raise it when the signal has a real noise floor -- each
        bucket folds in ~||tail||_2/sqrt(B) of noise and coefficients below
        that are not reliably located.
    loc_loops, est_loops : location loops and additional estimation loops.
        Location loops also contribute bucket values to estimation, so each
        candidate gets loc_loops + est_loops estimates.
    loop_threshold : votes needed out of loc_loops (default loc_loops//2 + 1).
    B_thresh : buckets kept per location loop (default 2*k, capped at B).
    tolerance, box_scale : filter sidelobe level and flat-top width in units
        of n/B.  Ignored when `filt` is supplied.
    gf_floor : estimates whose filter weight falls below this are dropped
        instead of divided through; the median is taken over the rest.
    max_candidates : safety cap on surviving candidates (default 10*k + 128);
        beyond it, the most-voted are kept.
    filt : a `Filter` from `flat_filter` to reuse, skipping construction.
    rng : seed or numpy Generator.
    batch : loops per batched gather; lower it to cap peak memory.

    Returns
    -------
    freqs : int array of recovered frequency indices, descending by magnitude.
    coeffs : complex array of estimates, scaled like numpy.fft.fft(x).
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
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
        filt = flat_filter(n, B, tolerance=tolerance, box_scale=box_scale,
                           dtype=x.dtype if x.dtype.kind == "c" else complex)
    h, Gf, M2, Wp = filt.h, filt.Gf, filt.M2, filt.Wp

    B_thresh = int(min(B_thresh if B_thresh is not None else 2 * k, B))
    if loop_threshold is None:
        loop_threshold = loc_loops // 2 + 1
    loop_threshold = int(min(max(loop_threshold, 1), loc_loops))
    if max_candidates is None:
        max_candidates = 10 * k + 128
    L = loc_loops + est_loops

    pow2 = (n & (n - 1)) == 0                    # then mod n is a bitmask
    bucket_width = n // B
    supp = np.arange(Wp, dtype=np.int64)
    step = np.arange(bucket_width, dtype=np.int64)

    sigmas = np.empty(L, dtype=np.int64)
    taus = np.empty(L, dtype=np.int64)
    for i in range(L):
        s = int(rng.integers(0, n))
        while gcd(s, n) != 1:                    # invertible mod n
            s = int(rng.integers(0, n))
        sigmas[i], taus[i] = s, int(rng.integers(0, n))

    # --- permute, filter, fold into B buckets, batched length-B FFTs ------
    Z = np.empty((L, B), dtype=complex)
    for lo in range(0, L, batch):
        hi = min(lo + batch, L)
        idx = sigmas[lo:hi, None] * supp[None, :] + taus[lo:hi, None]
        if pow2:
            idx &= n - 1
        else:
            idx %= n
        u = x[idx] * h
        Z[lo:hi] = np.fft.fft(u.reshape(hi - lo, Wp // B, B).sum(axis=1), axis=-1)

    # --- location: vote for the contents of the heaviest buckets -----------
    votes = np.zeros(n, dtype=np.uint8 if L < 255 else np.uint32)
    for i in range(loc_loops):
        sigma_inv = pow(int(sigmas[i]), -1, n)
        top = np.argpartition(-np.abs(Z[i]), B_thresh - 1)[:B_thresh]
        low = np.ceil((top - 0.5) * bucket_width).astype(np.int64)
        cand = sigma_inv * (low[:, None] + step[None, :]).ravel()
        if pow2:
            cand &= n - 1
        else:
            cand %= n
        votes[cand] += 1        # distinct within a loop: no scatter-add needed
    freqs = np.flatnonzero(votes >= loop_threshold)
    if freqs.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=complex)
    if freqs.size > max_candidates:
        keep = np.argpartition(-votes[freqs], max_candidates - 1)[:max_candidates]
        freqs = np.sort(freqs[keep])

    # --- estimation: invert the hash in every loop, then take medians ------
    perm = (sigmas[:, None] * freqs[None, :]) % n
    b = np.rint(perm / bucket_width).astype(np.int64) % B
    off = b * bucket_width - perm
    off -= n * np.rint(off / n).astype(np.int64)          # into (-n/2, n/2]
    weight = Gf[off + M2]
    ests = np.take_along_axis(Z, b, axis=1) / weight
    ests *= np.exp(-2j * np.pi * (taus[:, None] * freqs[None, :] % n) / n)
    ests[np.abs(weight) < gf_floor] = np.nan

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
# Demo / self-test / benchmark
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

    def best(fn, r=3):
        fn()
        out = []
        for _ in range(r):
            t = time.perf_counter()
            fn()
            out.append(time.perf_counter() - t)
        return min(out)

    print("accuracy and speed vs np.fft.fft, k = 50")
    print(f"{'n':>8} {'found':>8} {'rel err':>10} {'samples':>10} {'np.fft':>10}"
          f" {'sfft1':>10} {'speedup':>8}")
    for lg in (18, 20, 22, 24):
        n, k = 1 << lg, 50
        x, X, freqs = sparse_signal(n, k)
        f_hat, c_hat = sfft1(x, k, rng=1)
        t_np = best(lambda: np.fft.fft(x))
        t_s = best(lambda: sfft1(x, k, rng=1))
        found = len(set(f_hat.tolist()) & set(freqs.tolist()))
        err = np.max(np.abs(c_hat - X[f_hat])) / np.max(np.abs(X))
        B = min(1 << max(1, int(round(0.5 * np.log2(n * k / 5.0)))), n)
        used = 20 * flat_filter(n, B).W
        print(f"    2^{lg:<4d} {found:>4}/{k:<3} {err:>10.1e} {used / n:>9.2f}n"
              f" {t_np * 1e3:>9.1f}ms {t_s * 1e3:>9.1f}ms {t_np / t_s:>7.2f}x")

    print("\nspeed setting (tolerance=1e-4, est_loops=8), n = 2^24, k = 50")
    n, k = 1 << 24, 50
    x, X, freqs = sparse_signal(n, k)
    fl = flat_filter(n, 4096, tolerance=1e-4)
    f_hat, c_hat = sfft1(x, k, filt=fl, est_loops=8, rng=1)
    t_s = best(lambda: sfft1(x, k, filt=fl, est_loops=8, rng=1))
    t_np = best(lambda: np.fft.fft(x))
    print(f"  {len(set(f_hat.tolist()) & set(freqs.tolist()))}/{k} found, "
          f"rel err {np.max(np.abs(c_hat - X[f_hat])) / np.max(np.abs(X)):.1e}, "
          f"{t_s * 1e3:.1f} ms vs {t_np * 1e3:.0f} ms ({t_np / t_s:.1f}x), "
          f"reading {12 * fl.W / n:.3f}n samples")

    print("\ncrossover in sparsity, n = 2^22 (np.fft = "
          f"{best(lambda: np.fft.fft(np.zeros(1 << 22, complex))) * 1e3:.0f} ms)")
    n = 1 << 22
    t_np = best(lambda: np.fft.fft(np.zeros(n, complex)))
    for k in (10, 200, 1000, 4000):
        x, X, freqs = sparse_signal(n, k)
        B = min(1 << max(1, int(round(0.5 * np.log2(n * k / 5.0)))), n)
        fl = flat_filter(n, B, tolerance=1e-4)
        f_hat, _ = sfft1(x, k, filt=fl, est_loops=8, rng=1)
        t_s = best(lambda: sfft1(x, k, filt=fl, est_loops=8, rng=1))
        miss = k - len(set(f_hat.tolist()) & set(freqs.tolist()))
        print(f"  k = {k:>5}: {t_s * 1e3:>7.1f} ms  {t_np / t_s:>5.2f}x  missed {miss}")

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

    print("\nnoisy signals (n = 2^18, k = 40, default settings)")
    for snr in (30, 20, 10, 0):
        x, X, freqs = sparse_signal(1 << 18, 40, snr_db=snr)
        f_hat, c_hat = sfft1(x, 40, rng=2)
        found = len(set(f_hat.tolist()) & set(freqs.tolist()))
        rel = np.linalg.norm(c_hat - X[f_hat]) / np.linalg.norm(X[freqs])
        print(f"  SNR {snr:>3} dB: {found:>2}/40 located, relative error {rel:.4f}")

    print("\nreusing one filter over many signals (n = 2^20, k = 30)")
    n, k = 1 << 20, 30
    fl = flat_filter(n, 1 << 11)
    signals = [sparse_signal(n, k) for _ in range(20)]
    t, missed = time.perf_counter(), 0
    for s, (x, X, freqs) in enumerate(signals):
        f_hat, _ = sfft1(x, k, filt=fl, rng=s)
        missed += k - len(set(f_hat.tolist()) & set(freqs.tolist()))
    print(f"  20 transforms, {missed} of {20 * k} coefficients missed, "
          f"{(time.perf_counter() - t) / 20 * 1e3:.1f} ms each")
