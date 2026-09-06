"""
Sparse DFT by uniform decimation and binary phase encoding.

A streaming-compatible sparse Fourier transform, intended as a reference model
for an FPGA implementation.  Recovers the k largest coefficients of a length-n
spectrum while consuming only ``(log2(D)+1)/D`` of the input stream.

DFT convention matches numpy.fft:   X[m] = sum_i x[i] exp(-2j*pi*i*m/n)

------------------------------------------------------------------------------
How it works
------------------------------------------------------------------------------
Decimate the input by D (a power of two) and let B = n/D.  Branch `tau` reads
samples i = tau (mod D) only, so branches never contend for a sample and the
full-rate stream is never needed anywhere.  Writing S = sum(w) for a length-B
analysis window w, and scaling each branch transform by D*B/S:

    Y_tau[b] = sum_m X[m] * Wc[(b - m) mod B] * exp(2j*pi*tau*m/n) / Wc[0]

where Wc is the DFT of w.  Two consequences drive everything:

1. All m in one alias class (m = b mod B) share the same Wc[0] factor, and the
   window suppresses everything outside the class.  So a bin holding exactly
   one heavy coefficient satisfies |Y_tau[b]| = |X[m]| for *every* tau.  That
   equality is the singleton test, and it is what makes false positives rare:
   a bin holding two coefficients has tau-dependent magnitude, because the two
   contributions rotate at different rates and beat against each other.

2. For such a bin, Y_tau[b]/Y_0[b] = exp(2j*pi*tau*m/n) exactly -- the window
   contributes the same factor to every branch and cancels in the ratio.  In
   fact for a tone at any real frequency f0, on-grid or not, the tau
   dependence of *every* bin is exactly exp(2j*pi*f0*tau/n), independent of b
   and of the window.  So leakage never corrupts the phase relation; it only
   adds other tones' energy to a bin.  The window is therefore about dynamic
   range, not correctness -- see `make_plan`.

Aliasing already reveals m mod B, which is log2(B) of the log2(n) bits of m.
The remaining log2(D) bits, m = b + B*q, come from the phase ratios.  With the
geometric ladder tau = D/2, D/4, ..., 1, at step t (having already determined
bits 0..t-1 of q as q_partial):

    R = Y_tau[b]/Y_0[b] * exp(-2j*pi*tau*(b + B*q_partial)/n) = exp(1j*pi*bit_t)

The bits above t drop out because they contribute whole multiples of 2*pi.  So
every bit is a sign decision on a de-rotated ratio with a full +-1 of margin,
independent of the bits not yet known.  No arctan, no CORDIC, and no divider is
needed -- compare cross products instead of forming the ratio.

------------------------------------------------------------------------------
Notes for a hardware implementation
------------------------------------------------------------------------------
* One branch = one B-point FFT.  For n = 2^20 and D = 64 that is 7 FFTs of
  16384 points, all within vendor streaming-FFT IP range, versus a dense 2^20
  transform that exceeds it and forces a four-step decomposition with a
  full-record transpose.
* Branches read fixed residues mod D, so they can be fed by D-way
  time-interleaved converters at f_s/D each -- the full-rate stream never
  crosses into the fabric.
* The de-rotation factors out as exp(-2j*pi*tau*b/n) * exp(-2j*pi*q/2^(t+1)).
  The first is a per-(branch, bin) constant that can be folded into the FFT
  output twiddle; the second takes only 2^(t+1) distinct values, i.e. a LUT of
  at most D entries.  See `derotation_tables`.
* Bins failing the singleton test are reported in `unresolved` rather than
  guessed at.  With a power-of-two D the residues of different D are nested,
  so a second decimation stage cannot separate a colliding pair and peeling
  does not help: two coefficients congruent mod B collide on every frame.
  Raise B, or move n off a power of two for genuinely co-prime moduli.
* Bin magnitudes fold D coefficients' worth of noise, so the usable threshold
  scales as ||tail||_2/sqrt(B).  Choose D from the SNR, not the LUT budget.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "Plan", "make_plan", "derotation_tables", "StreamingFrontEnd",
    "decode", "sparse_dft", "quantize",
]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
class Plan(NamedTuple):
    """Fixed geometry and coefficients; depends only on (n, D, window)."""
    n: int
    D: int
    B: int
    taus: np.ndarray        # branch shifts: [0, D/2, D/4, ..., 1]
    window: np.ndarray      # length-B analysis window, identical on all branches
    scale: float            # normalisation so Y[0, b] ~ X[m]
    merge_radius: int       # bins one off-grid tone can light up, one side


def make_plan(n, D, window="kaiser", beta=12.0, merge_radius=None) -> Plan:
    """Build the geometry for a length-n transform decimated by D.

    `window` is 'rect' or 'kaiser'.  Which to use is decided by dynamic range,
    not by whether the signal is on-grid, and the measured differences are:

      signal                          rect      kaiser b=12
      on-grid                        99.3/100   96.8/100
      off-grid, equal amplitudes     99.0/100   98.2/100
      off-grid, 40 dB spread         89.2/100   99.8/100

    Rectangular is exact for on-grid coefficients (Wc is a delta, so there is
    no leakage at all) and costs no multiplier, and it stays fine off-grid as
    long as amplitudes are comparable.  What breaks it is dynamic range: an
    off-grid tone leaks with only 1/offset decay, so a strong emitter buries
    weak ones several thousand bins away.  Kaiser confines each tone to a few
    adjacent bins, at the cost of smearing on-grid coefficients across a main
    lobe -- which is why it scores *worse* on the on-grid row.  For a real
    receiver front end, expect off-grid tones and wide dynamic range, so
    kaiser is the default.
    """
    if n % D:
        raise ValueError(f"D={D} must divide n={n}")
    if D & (D - 1):
        raise ValueError(f"D={D} must be a power of two")
    B = n // D
    T = int(np.log2(D))
    taus = np.array([0] + [D >> (t + 1) for t in range(T)], dtype=np.int64)
    if len(set(taus.tolist())) != len(taus):
        raise ValueError("branch shifts collide; D too small")

    if window == "rect":
        w = np.ones(B)
        mr = 1
    elif window == "kaiser":
        w = np.kaiser(B, beta)
        mr = int(round(np.sqrt(1 + (beta/np.pi)**2))) + 1   # main-lobe half width
    else:
        w = np.asarray(window, dtype=float)
        if w.size != B:
            raise ValueError(f"window must have length B={B}")
        mr = 3
    return Plan(n, D, B, taus, w, D * B / w.sum(),
                mr if merge_radius is None else merge_radius)


def derotation_tables(plan: Plan):
    """The de-rotation constants a hardware decoder would store.

    Returns (per_bin, per_q) where the factor needed at bit t for bin b with
    partial quotient q is ``per_bin[t][b] * per_q[t][q & (2**(t+1) - 1)]``.
    per_bin is (T, B) of per-(branch, bin) constants that can be folded into
    the FFT output stage; per_q is a ragged list of small LUTs, 2^(t+1) entries
    at bit t, so at most D entries in total across all bits.
    """
    n, B, taus = plan.n, plan.B, plan.taus
    T = len(taus) - 1
    per_bin = np.stack([np.exp(-2j*np.pi*taus[t+1]*np.arange(B)/n) for t in range(T)])
    per_q = [np.exp(-2j*np.pi*np.arange(1 << (t + 1)) / (1 << (t + 1)))
             for t in range(T)]
    return per_bin, per_q


# ---------------------------------------------------------------------------
# Streaming front end
# ---------------------------------------------------------------------------
class StreamingFrontEnd:
    """Routes an arriving sample stream into per-branch buffers.

    Mirrors the hardware dataflow: samples arrive in index order, and the
    sample at index i is consumed by the single branch whose shift satisfies
    tau = i (mod D), or discarded.  `samples_kept` and `samples_seen` let you
    confirm the duty cycle.  In hardware each branch's FFT would consume its
    samples as they arrive at f_s/D rather than filling a buffer first; the
    two are equivalent, buffering here only keeps the model simple.
    """

    def __init__(self, plan: Plan, dtype=complex):
        self.plan = plan
        self.dtype = dtype
        # residue -> branch index, or -1 for "discard"
        self.route = np.full(plan.D, -1, dtype=np.int64)
        self.route[plan.taus] = np.arange(len(plan.taus))
        self.reset()

    def reset(self):
        p = self.plan
        self.buf = np.zeros((len(p.taus), p.B), dtype=self.dtype)
        self.samples_seen = 0
        self.samples_kept = 0

    def push(self, chunk):
        """Feed the next samples in arrival order.  Returns True if the frame
        is complete."""
        p = self.plan
        chunk = np.asarray(chunk)
        i = self.samples_seen + np.arange(chunk.size, dtype=np.int64)
        keep = i < p.n
        i, chunk = i[keep], chunk[keep]
        br = self.route[i & (p.D - 1)]
        sel = br >= 0
        self.buf[br[sel], i[sel] // p.D] = chunk[sel]
        self.samples_seen += int(keep.sum())
        self.samples_kept += int(sel.sum())
        return self.samples_seen >= p.n

    def transforms(self):
        """The B-point branch transforms, normalised so bins carry X directly."""
        p = self.plan
        return p.scale * np.fft.fft(self.buf * p.window, axis=-1)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
def decode(Y, plan: Plan, threshold=None, rel_tol=0.20, noise_factor=6.0,
           rel_floor=1e-4, merge=True):
    """Detect singleton bins and decode their frequencies.

    Parameters
    ----------
    Y : (branches, B) branch transforms from `StreamingFrontEnd.transforms`.
    threshold : absolute magnitude floor on |Y[0]|.  Default
        ``max(noise_factor * median|Y[0]|, rel_floor * max|Y[0]|)``.  The
        median term is a robust noise-floor estimate that maps directly to a
        running-percentile threshold in hardware; the rel_floor term keeps a
        near-empty spectrum from promoting arithmetic dust into candidates, and
        also sets the dynamic range below the strongest bin that is searched.
    rel_tol : allowed relative spread of |Y[tau, b]| across branches for a bin
        to count as a singleton.  Tightening it rejects more collisions at the
        cost of rejecting noisy true singletons.
    merge : collapse detections in adjacent bins, which one off-grid tone
        produces, keeping the strongest.

    Returns
    -------
    freqs : recovered frequency indices, descending by magnitude.
    coeffs : corresponding coefficient estimates, scaled like numpy.fft.fft.
    unresolved : bins that cleared the threshold but failed the singleton
        test -- these are the blind spots, each standing for two or more
        coefficients somewhere in its alias class.
    """
    n, D, B, taus = plan.n, plan.D, plan.B, plan.taus
    T = len(taus) - 1
    mag = np.abs(Y)
    ref = mag[0]
    if threshold is None:
        threshold = max(noise_factor * np.median(ref), rel_floor * ref.max())

    loud = ref > threshold
    spread = (mag.max(axis=0) - mag.min(axis=0)) / np.maximum(ref, 1e-30)
    single = loud & (spread <= rel_tol)
    unresolved = np.flatnonzero(loud & ~single)

    bins = np.flatnonzero(single)
    if bins.size == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=complex), unresolved)

    # bit-by-bit decode, vectorised over all candidate bins at once
    q = np.zeros(bins.size, dtype=np.int64)
    ratio = Y[1:, bins] / Y[0, bins]
    ok = np.ones(bins.size, dtype=bool)
    for t in range(T):
        R = ratio[t] * np.exp(-2j*np.pi*taus[t+1]*(bins + B*q)/n)
        ok &= np.abs(np.abs(R) - 1.0) <= 3*rel_tol
        q |= (R.real < 0).astype(np.int64) << t

    freqs = bins + B*q
    keep = ok & (q < D) & (freqs < n)
    freqs, vals, src = freqs[keep], Y[0, bins[keep]], bins[keep]

    if merge and freqs.size:
        order = np.argsort(freqs)
        freqs, vals = freqs[order], vals[order]
        groups = np.concatenate(([0], np.cumsum(np.diff(freqs) > plan.merge_radius)))
        pick = np.zeros(groups[-1] + 1, dtype=np.int64)
        for g in range(groups[-1] + 1):
            idx = np.flatnonzero(groups == g)
            pick[g] = idx[np.argmax(np.abs(vals[idx]))]
        freqs, vals = freqs[pick], vals[pick]

    order = np.argsort(-np.abs(vals))
    return freqs[order], vals[order], unresolved


def sparse_dft(x, D, window="kaiser", beta=12.0, chunk=1 << 16, **kw):
    """Convenience wrapper: stream `x` through the front end and decode."""
    x = np.asarray(x)
    plan = make_plan(x.size, D, window=window, beta=beta)
    fe = StreamingFrontEnd(plan, dtype=complex)
    for lo in range(0, x.size, chunk):
        fe.push(x[lo:lo + chunk])
    return decode(fe.transforms(), plan, **kw) + (fe,)


# ---------------------------------------------------------------------------
# Fixed-point helper
# ---------------------------------------------------------------------------
def quantize(a, bits, peak=None):
    """Round to a signed `bits`-wide fixed-point grid with saturation.

    This bounds the contribution of *input and coefficient* quantization only.
    It is not a bit-accurate model: the FFT and decoder arithmetic below stay
    in float, so treat the result as an optimistic bound on wordlength.
    """
    if peak is None:
        peak = np.abs(a).max()
    if peak == 0:
        return a
    lsb = peak / (2 ** (bits - 1) - 1)
    if np.iscomplexobj(a):
        r = np.clip(np.round(a.real / lsb), -(2**(bits-1)), 2**(bits-1) - 1)
        i = np.clip(np.round(a.imag / lsb), -(2**(bits-1)), 2**(bits-1) - 1)
        return (r + 1j*i) * lsb
    return np.clip(np.round(a / lsb), -(2**(bits-1)), 2**(bits-1) - 1) * lsb


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, k = 1 << 20, 100

    def on_grid(nn, kk, snr_db=None, dyn_db=0.0):
        """Exactly DFT-sparse: every coefficient sits on an integer bin."""
        X = np.zeros(nn, dtype=complex)
        f = rng.choice(nn, kk, replace=False)
        X[f] = 10**(rng.random(kk)*dyn_db/20.0) * np.exp(2j*np.pi*rng.random(kk))
        x = np.fft.ifft(X)
        if snr_db is not None:
            s = np.sqrt(np.mean(np.abs(x)**2) / (2*10**(snr_db/10)))
            x = x + s*(rng.standard_normal(nn) + 1j*rng.standard_normal(nn))
        return x, X, f

    def off_grid(nn, kk, dyn_db=0.0, snr_db=None):
        """Tones at fractional frequencies, as a real receiver would see."""
        f = rng.choice(nn - 2, kk, replace=False) + rng.random(kk)
        a = 10**(rng.random(kk)*dyn_db/20.0) * np.exp(2j*np.pi*rng.random(kk))
        i = np.arange(nn)
        x = np.zeros(nn, dtype=complex)
        for fc, ac in zip(f, a):
            x += (ac/nn) * np.exp(2j*np.pi*fc*i/nn)
        if snr_db is not None:
            s = np.sqrt(np.mean(np.abs(x)**2) / (2*10**(snr_db/10)))
            x = x + s*(rng.standard_normal(nn) + 1j*rng.standard_normal(nn))
        return x, f, a

    def score_on(fr, f):
        t = set(f.tolist())
        return len(set(fr.tolist()) & t), len(set(fr.tolist()) - t)

    def score_off(fr, f):
        if fr.size == 0:
            return 0, 0
        near = np.abs(fr[:, None] - f[None, :]).min(axis=1) <= 1.0
        return int(near.sum()), int((~near).sum())

    R = 6
    print(f"n = 2^20, k = {k}, {R} trials per row\n")

    print("on-grid coefficients, rectangular window")
    print(f"{'D':>4} {'B':>7} {'br':>3} {'duty':>7} {'found':>10} {'false':>7}"
          f" {'unres':>7} {'rel err':>9}")
    for D in (256, 64, 32):
        g = fp = u = 0
        errs = []
        for _ in range(R):
            x, X, f = on_grid(n, k)
            fr, co, un = sparse_dft(x, D, window="rect")[:3]
            a, b = score_on(fr, f); g += a; fp += b; u += un.size
            hit = [m for m in fr.tolist() if X[m] != 0]
            if hit:
                d = dict(zip(fr.tolist(), co))
                errs.append(np.median([abs(d[m]-X[m])/abs(X[m]) for m in hit]))
        T = int(np.log2(D)) + 1
        print(f"{D:>4} {n//D:>7} {T:>3} {T/D*100:>6.1f}% {g/R:>7.1f}/{k} {fp/R:>7.1f}"
              f" {u/R:>7.1f} {np.median(errs):>9.1e}")

    print("\nwindow choice is set by dynamic range, not by on-grid vs off-grid")
    print(f"{'signal':>28} {'rect':>12} {'kaiser b=12':>13}")
    cases = [("on-grid", lambda: on_grid(n, k), score_on, 1),
             ("off-grid, equal amplitude", lambda: off_grid(n, k), score_off, 0),
             ("off-grid, 40 dB spread", lambda: off_grid(n, k, dyn_db=40), score_off, 0),
             ("on-grid, 40 dB spread", lambda: on_grid(n, k, dyn_db=40), score_on, 1)]
    for name, gen, scorer, idx in cases:
        cells = []
        for win, beta in (("rect", 0.0), ("kaiser", 12.0)):
            g = fp = 0
            for _ in range(R):
                out = gen()
                x = out[0]; truth = out[2] if idx else out[1]
                fr, co, un = sparse_dft(x, 64, window=win, beta=beta)[:3]
                a, b = scorer(fr, truth); g += a; fp += b
            cells.append(f"{g/R:5.1f}/{k} f{fp/R:4.1f}")
        print(f"{name:>28} {cells[0]:>12} {cells[1]:>13}")

    print("\nadditive noise (bins fold D coefficients' worth of it)")
    print(f"{'D':>4} {'window':>10} " + " ".join(f"{s:>14}" for s in
          ("noiseless", "SNR 20 dB", "SNR 10 dB", "SNR 0 dB")))
    for D, win, beta in ((256, "rect", 0.0), (64, "rect", 0.0), (64, "kaiser", 12.0)):
        cells = []
        for snr in (None, 20, 10, 0):
            g = fp = 0
            for _ in range(4):
                x, X, f = on_grid(n, k, snr_db=snr)
                fr, co, un = sparse_dft(x, D, window=win, beta=beta)[:3]
                a, b = score_on(fr, f); g += a; fp += b
            cells.append(f"{g/4:5.1f}/{k} f{fp/4:4.1f}")
        label = win if win == "rect" else f"kaiser {beta:.0f}"
        print(f"{D:>4} {label:>10} " + " ".join(f"{c:>14}" for c in cells))

    print("\ninput wordlength, off-grid + 40 dB spread, D = 64, kaiser b=12")
    print("  (input and window quantized; FFT and decoder still float, so this")
    print("   bounds the input-quantization term only, not the full datapath)")
    for bits in (8, 10, 12, 14, 16):
        g = fp = 0
        for _ in range(4):
            x, f, a = off_grid(n, k, dyn_db=40)
            plan = make_plan(n, 64)
            fe = StreamingFrontEnd(plan)
            fe.push(quantize(x, bits))
            Y = plan.scale * np.fft.fft(fe.buf * quantize(plan.window, 18), axis=-1)
            fr, co, un = decode(Y, plan)
            c, d = score_off(fr, f); g += c; fp += d
        print(f"  {bits:>2} bit: {g/4:>5.1f}/{k} located, {fp/4:>4.1f} false")
    print(f"  no trend: a {1<<14}-point FFT gives ~{10*np.log10(1<<14):.0f} dB of"
          " processing gain, so at these\n  parameters detection is limited by"
          " leakage and collisions, not by input bits.")

    print("\nstream duty and on-chip cost, n = 2^20")
    x, X, f = on_grid(n, k)
    for D in (256, 64, 32):
        fr, co, un, fe = sparse_dft(x, D, window="rect")
        p = fe.plan
        print(f"  D = {D:>3}: kept {fe.samples_kept:>7} of {fe.samples_seen}"
              f" ({100*fe.samples_kept/fe.samples_seen:>5.2f}%), "
              f"{len(p.taus)} x {p.B}-pt FFT, "
              f"{len(p.taus)*p.B*32/1e6:>4.2f} Mbit @16-bit I/Q, "
              f"{len(p.taus)*p.B*np.log2(p.B)/1e6:.2f}M butterflies")
    print(f"  dense: needs all {n} samples, ~{n*32/1e6:.1f} Mbit, "
          f"{n*np.log2(n)/1e6:.1f}M butterflies")

    pb, pq = derotation_tables(make_plan(n, 64))
    print(f"\n  de-rotation at D=64: per-bin table {pb.shape} (foldable into the FFT"
          f" output twiddles)\n  plus per-q LUTs of {[len(t) for t in pq]}"
          f" = {sum(len(t) for t in pq)} entries total")

    # streaming must equal batch regardless of how the stream is chopped up
    x, X, f = on_grid(n, k)
    plan = make_plan(n, 64, window="rect")
    ref = None
    for ch in (1 << 20, 1 << 13, 1000, 7):
        fe = StreamingFrontEnd(plan)
        for lo in range(0, n, ch):
            fe.push(x[lo:lo + ch])
        fr, co, un = decode(fe.transforms(), plan)
        if ref is None:
            ref = (fr, co)
        assert np.array_equal(fr, ref[0]) and np.allclose(co, ref[1])
    print("\n  streaming result independent of chunk size: OK")
