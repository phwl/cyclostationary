"""
R-FFAST -- robust sub-linear-time sparse DFT via a fixed, multi-stage
aliasing front end and an onion-peeling back end.

Implementation of

    S. Pawar and K. Ramchandran, "A robust sub-linear time R-FFAST
    algorithm for computing a sparse DFT", arXiv:1501.00320 (2015).

built as a drop-in `s3ca.py` backend (see `RFFASTBackend` at the bottom),
exposing the same ``required_indices(seed)`` / ``run(x_col, kappa, seed)``
contract as `sfft_opt._SFFT1Backend` and `decimated_sfft._DecimatedBackend`.

DFT convention matches numpy.fft:  X[m] = sum_p x[p] exp(-2j*pi*p*m/n).

------------------------------------------------------------------------------
How it works (Sections V, VII, VIII of the paper)
------------------------------------------------------------------------------
Front end: d >= 3 "stages". Stage s aliases the spectrum into f[s] bins by
subsampling x with period P[s] = n // f[s]:

    Y_s,tau[b] = P[s] * FFT_{f[s]}( x[(tau + m*P[s]) mod n], m=0..f[s]-1 )[b]
               = sum_{l == b (mod f[s])} X[l] * exp(i*2*pi*l*tau/n)

for every delay tau in a shared set of D = C*Nc delays, reused identically
across every stage (Section VII, "a convenience that is sufficient, and
makes the R-FFAST framework implementation friendly"). The delays are
grouped into C "clusters" of Nc samples each; cluster i's samples are
spaced 2**i apart starting from a random offset r_i (Section VII-B.2):

    tau(i, j) = (r_i + j * 2**i) mod n,   j = 0 .. Nc-1

This geometric spacing is what lets the back end recover l one bit at a
time (Section VI/VIII): viewed as a function of tau, a singleton bin's
observation is a single complex sinusoid of angular frequency
omega = 2*pi*l/n, so cluster i's samples (spaced 2**i apart in tau) see a
phase increment of 2**i * omega per step -- a coarser, more wrapped view
for larger i. Fusing all C clusters via successive refinement narrows the
ambiguity in omega from a full cycle (cluster 0) down to ~2*pi/n (cluster
C-1), i.e. single-bin resolution -- with no arctan/CORDIC per bit, and no
per-stage filter (unlike SFFT 2.0's Dolph-Chebyshev bandpass): this is the
"aliasing-based" family, not "subsampling-based" (see the CS-matrix-design
letter's Table I / Section II-A for that distinction).

Back end: for each bin in each stage, test its energy across the D delays
against a noise threshold (zero-ton), else estimate its frequency and fit
an amplitude (singleton test via residual energy), else leave it as an
unresolved multi-ton. Every accepted singleton is "peeled": its estimated
contribution is subtracted from the corresponding bin in every stage,
which can reveal new singletons there. Iterate to convergence.

------------------------------------------------------------------------------
A real compatibility constraint (flagged up front, not glossed over)
------------------------------------------------------------------------------
The very-sparse-regime front end (Section VII-A) needs `d` pairwise-coprime
divisors of n, each near kappa. A power-of-two n admits at most *one*
divisor greater than 1 that is coprime to all the others (any two divisors
of a power of two share a factor of 2), so `n` must be chosen with this
factorization in mind -- e.g. the paper's own example, n = 100*101*103*99,
kappa = 100. `choose_stage_bins` finds such a factorization automatically
when one exists and raises a clear `ValueError` (with a usable suggestion)
when it does not, rather than silently degrading.

------------------------------------------------------------------------------
Deliberate simplifications, stated explicitly
------------------------------------------------------------------------------
* Noise threshold. The paper's T = (1+gamma)*D assumes a known, unit-
  variance-normalised noise model (Section II). Here `noise_var=None`
  falls back to a percentile-based estimate from the observed bin
  energies pooled across all stages -- adequate for a first pass, but a
  known noise variance (pass it directly) reproduces the paper's own
  analysis and is preferred when available. This mirrors the analogous
  heuristic-vs-known-SNR choice already made in `decimated_sfft.py`.
* Kay's (1989) exact MMSE weights for the phase-difference estimator
  (eq. 11) are reproduced from the standard closed form and then
  re-normalised to sum to 1; the renormalisation makes the estimator
  unbiased regardless of any transcription slip in the closed form, at
  the cost of at most a small efficiency loss relative to the literature
  value.
* Peeling is batched per (stage, pass) rather than strictly bin-by-bin
  (Algorithm 1 in the paper is written as a triple nested loop over
  iteration/stage/bin, sequentially). Within one stage's pass, a bin
  freed up by another bin peeled earlier in the *same* pass isn't caught
  until the next outer iteration. This trades a small amount of extra
  `iterations` for large speedups from vectorising over bins with numpy,
  matching the style already used in `decimated_sfft.decode`.
"""

from __future__ import annotations

from math import gcd
from typing import NamedTuple

import numpy as np

__all__ = [
    "Plan", "make_plan", "choose_stage_bins", "kay_weights",
    "compute_front_end", "front_end_indices", "estimate_frequency",
    "peel_decode", "RFFASTBackend",
]


# ---------------------------------------------------------------------------
# Stage-bin selection (Section VII-A, "very-sparse regime")
# ---------------------------------------------------------------------------
def _divisors(n):
    """All divisors of n, via trial division to sqrt(n)."""
    divs = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
        i += 1
    return sorted(divs)


def choose_stage_bins(n, kappa, d=3):
    """Find `d` pairwise-coprime divisors of `n`, each as close to `kappa`
    as available divisors allow (Section VII-A's construction: f_i =
    O(kappa) + O(1), relatively coprime factors of n).

    Raises ValueError, with a concrete suggestion, if no such factorization
    exists -- most commonly because `n` is a power of two.
    """
    divs = [x for x in _divisors(n) if x > 1]
    divs.sort(key=lambda x: abs(x - kappa))
    chosen: list[int] = []
    for x in divs:
        if all(gcd(x, c) == 1 for c in chosen):
            chosen.append(x)
            if len(chosen) == d:
                return np.array(sorted(chosen), dtype=np.int64)
    raise ValueError(
        f"could not find {d} pairwise-coprime divisors of n={n} near "
        f"kappa={kappa}; R-FFAST's front end needs n to factor into d "
        f"near-kappa, pairwise-coprime pieces (e.g. n = 100*101*103*99 for "
        f"kappa=100, per the paper's own example). A pure power-of-two n "
        f"cannot supply more than one such divisor. Either pick a highly "
        f"composite n, or pass an explicit f=[...] list of pairwise-coprime "
        f"divisors of n to make_plan()."
    )


# ---------------------------------------------------------------------------
# Plan: fixed front-end geometry (depends only on n, kappa, and the seed
# used to draw cluster offsets -- not on any particular signal)
# ---------------------------------------------------------------------------
class Plan(NamedTuple):
    n: int
    f: np.ndarray          # (d,) stage bin counts, pairwise-coprime divisors of n
    shifts: np.ndarray     # (C, Nc) int64 delays: shifts[i,j] = (r_i + j*2**i) mod n


def make_plan(n, kappa=None, f=None, d=3, clusters=None, per_cluster=3,
              seed=0) -> Plan:
    """Build the fixed R-FFAST front-end geometry for transforms of length n.

    Either pass `kappa` (auto-selects `d` pairwise-coprime stage-bin counts
    near kappa via `choose_stage_bins`) or pass `f` directly (validated:
    must divide n and be pairwise coprime).

    `clusters` (C) defaults to enough clusters that the successive-refinement
    ladder's final resolution, ~2*pi/2**(C-1), is finer than one DFT bin
    (2*pi/n) with margin -- i.e. C ~ log2(n) + a few (Section VII-B.2,
    "C = O(log n)"). `per_cluster` (Nc) defaults to 3, the paper's own
    empirical finding (Section IX-A) rather than its asymptotic O(log^(1/3) n)
    bound, which is far too conservative at the sizes used here.
    """
    if f is None:
        if kappa is None:
            raise ValueError("pass either kappa (to auto-select f) or f directly")
        f = choose_stage_bins(n, kappa, d=d)
    else:
        f = np.asarray(f, dtype=np.int64)
        for x in f:
            if n % int(x):
                raise ValueError(f"stage bin count {x} does not divide n={n}")
        for a in range(len(f)):
            for b in range(a + 1, len(f)):
                if gcd(int(f[a]), int(f[b])) != 1:
                    raise ValueError(
                        f"stage bin counts must be pairwise coprime; "
                        f"gcd({f[a]}, {f[b]}) != 1")

    if clusters is None:
        clusters = max(6, int(np.ceil(np.log2(max(n, 2)))) + 4)
    if per_cluster < 2:
        raise ValueError("per_cluster (Nc) must be >= 2")
    if (1 << (clusters - 1)) < n:
        raise ValueError(
            f"clusters={clusters} gives a final resolution ladder of only "
            f"2**{clusters-1}, too coarse to disambiguate n={n} candidate "
            f"frequencies; raise `clusters`.")

    rng = np.random.default_rng(seed)
    r = rng.integers(0, n, size=clusters)
    j = np.arange(per_cluster)
    shifts = (r[:, None] + j[None, :] * (1 << np.arange(clusters))[:, None]) % n
    return Plan(n, f, shifts.astype(np.int64))


# ---------------------------------------------------------------------------
# Kay's (1989) weighted phase-difference frequency estimator
# ---------------------------------------------------------------------------
def kay_weights(Nc):
    """MMSE weights beta(t), t = 0..Nc-2, for averaging Nc-1 consecutive
    phase differences into one frequency estimate (paper's eq. 11, closed
    form from S. Kay, "A fast and accurate single frequency estimator",
    IEEE ASSP 1989). Renormalised to sum to 1 (keeps the estimator unbiased
    regardless of any transcription slip in the closed form -- see module
    docstring)."""
    t = np.arange(Nc - 1, dtype=float)
    w = 1.0 - ((2 * t - Nc + 2) / Nc) ** 2
    w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(Nc - 1, 1.0 / (Nc - 1))


def _kay_omega(y, beta):
    """y: (..., Nc) complex samples equally spaced by one unit of the
    *local* index (the caller is responsible for descaling by the actual
    physical spacing, e.g. 2**i for cluster i). Returns the raw phase-
    increment-per-step estimate, wrapped to (-pi, pi]."""
    diffs = np.angle(y[..., 1:] * np.conj(y[..., :-1]))
    return np.sum(diffs * beta, axis=-1)


def estimate_frequency(Y_clusters, beta, n):
    """Successive-refinement frequency estimate (Algorithm 2, steps 3-16).

    Y_clusters : (nb, C, Nc) complex, one candidate bin's observation
        reshaped into C clusters of Nc samples each -- see `Plan.shifts`.
    beta : (Nc-1,) weights from `kay_weights`.
    n : transform length (only used to report a wrapped omega; the caller
        maps omega -> a support estimate ell = round(omega*n/2pi) mod n).

    Returns omega_hat, shape (nb,), in [0, 2*pi).
    """
    nb, C, Nc = Y_clusters.shape
    omega_prev = np.zeros(nb)
    for i in range(C):
        raw = _kay_omega(Y_clusters[:, i, :], beta)     # estimates (2**i * omega) mod 2pi
        omega_new = raw / (1 << i)                       # -> omega, ambiguous mod 2pi/2**i
        period = 2 * np.pi / (1 << i)
        d1 = np.ceil(omega_prev / period) * period + omega_new - omega_prev
        d2 = np.floor(omega_prev / period) * period + omega_new - omega_prev
        delta = np.where(np.abs(d1) < np.abs(d2), d1, d2)
        omega_prev = omega_prev + delta
    return omega_prev % (2 * np.pi)


# ---------------------------------------------------------------------------
# Front end: compute every stage's aliased bin observations
# ---------------------------------------------------------------------------
def compute_front_end(x, plan: Plan):
    """Returns a list of length d, one (C, Nc, f[s]) complex array per
    stage: Ys[s][i, j, b] = Y_{s, shifts[i,j]}[b] (see module docstring)."""
    n = plan.n
    C, Nc = plan.shifts.shape
    taus = plan.shifts.reshape(-1)
    Ys = []
    for f in plan.f:
        f = int(f)
        P = n // f
        m = np.arange(f, dtype=np.int64)
        idx = (taus[:, None] + m[None, :] * P) % n         # (C*Nc, f)
        v = x[idx]
        Y = P * np.fft.fft(v, axis=-1)                      # (C*Nc, f)
        Ys.append(Y.reshape(C, Nc, f))
    return Ys


def front_end_indices(plan: Plan):
    """The distinct raw x[] positions the front end reads -- this is what
    `s3ca.py`'s COMPIDX-style restriction (mode="full") uses as W'. Fixed
    once `plan` is built: independent of the signal, by design (Section
    VII's "fixed sampling structure" -- see module docstring)."""
    n = plan.n
    taus = plan.shifts.reshape(-1)
    idxs = []
    for f in plan.f:
        f = int(f)
        P = n // f
        m = np.arange(f, dtype=np.int64)
        idxs.append(((taus[:, None] + m[None, :] * P) % n).ravel())
    return np.unique(np.concatenate(idxs))


# ---------------------------------------------------------------------------
# Back end: onion-peeling decoder (Algorithm 1)
# ---------------------------------------------------------------------------
def peel_decode(Ys, plan: Plan, beta, kappa=None, iterations=8,
                 noise_var=None, gamma=0.5, energy_percentile=20.0,
                 resid_slack=0.5):
    """Peeling decode over the front-end output `Ys` (mutated in place).

    Parameters
    ----------
    Ys : list of (C, Nc, f[s]) arrays, as returned by `compute_front_end`.
    plan, beta : as above.
    kappa : cap the number of returned coefficients to the kappa strongest
        (by |value|); None returns everything decoded.
    iterations : max number of full sweeps over all stages/bins. Peeling
        iterations are constant in the paper's own analysis (Appendix C);
        a small fixed budget suffices in practice.
    noise_var : per-sample noise variance, if known. When given, the
        threshold is exactly the paper's T = (1+gamma)*D*noise_var
        (D = C*Nc samples per bin). When None (default), falls back to a
        percentile-based threshold over the pooled per-bin energies from
        all stages (see module docstring for the tradeoff).
    resid_slack : the residual-energy singleton/multi-ton cutoff is
        threshold*(1+resid_slack), matching Algorithm 2 step 19's T but
        with a little extra margin since the fallback threshold above is
        already an approximation.

    Returns
    -------
    freqs, coeffs : int64 and complex128 arrays, sorted by descending |coeff|.
    """
    n = plan.n
    C, Nc = plan.shifts.shape
    D = C * Nc
    taus = plan.shifts.reshape(-1)
    fs = [int(f) for f in plan.f]
    Ps = [n // f for f in fs]          # per-stage subsampling period P_s = n/f_s

    # Per-stage noise threshold. Subsampling by P_s and rescaling by P_s to
    # match X's amplitude (see compute_front_end / module docstring) inflates
    # a raw per-sample noise variance sigma^2 to P_s*n*sigma^2 per (bin,
    # delay) -- an aliasing/decimation gain loss that differs by stage since
    # P_s = n/f_s does. Summed over D delays: T_s = (1+gamma)*D*P_s*n*sigma^2.
    # When sigma^2 (noise_var) isn't known, fall back to a per-stage
    # percentile of that stage's own bin energies -- pooling across stages
    # here would be wrong, since each stage's noise floor sits at a
    # different scale.
    if noise_var is not None:
        thresh = [(1.0 + gamma) * D * P * n * noise_var for P in Ps]
    else:
        thresh = [max(np.percentile((np.abs(Y) ** 2).sum(axis=(0, 1)),
                                     energy_percentile), 1e-300)
                  for Y in Ys]

    decoded: dict[int, complex] = {}

    for _ in range(iterations):
        progress = False
        for s, f in enumerate(fs):
            Y = Ys[s]
            T = thresh[s]
            energy = (np.abs(Y) ** 2).sum(axis=(0, 1))       # (f,)
            loud = np.flatnonzero(energy > T)
            if loud.size == 0:
                continue

            Yb = Y[:, :, loud].transpose(2, 0, 1)             # (nb, C, Nc)
            omega = estimate_frequency(Yb, beta, n)
            ell = np.round(omega * n / (2 * np.pi)).astype(np.int64) % n

            steer = np.exp(1j * 2 * np.pi * np.outer(ell, taus) / n)  # (nb, D)
            yb_flat = Yb.reshape(loud.size, D)
            val = (np.conj(steer) * yb_flat).sum(axis=1) / D
            resid = yb_flat - val[:, None] * steer
            resid_energy = (np.abs(resid) ** 2).sum(axis=1)
            is_single = resid_energy < T * (1.0 + resid_slack)

            for bi in np.flatnonzero(is_single):
                L = int(ell[bi])
                if L in decoded:
                    continue
                v = complex(val[bi])
                decoded[L] = v
                progress = True
                phase = np.exp(1j * 2 * np.pi * L * taus / n).reshape(C, Nc)
                for s2, f2 in enumerate(fs):
                    b2 = L % f2
                    Ys[s2][:, :, b2] -= v * phase
        if not progress:
            break

    order = sorted(decoded, key=lambda k: -abs(decoded[k]))
    freqs = np.array(order, dtype=np.int64)
    coeffs = np.array([decoded[k] for k in order], dtype=complex)
    if kappa is not None and freqs.size > kappa:
        freqs, coeffs = freqs[:kappa], coeffs[:kappa]
    return freqs, coeffs


# ---------------------------------------------------------------------------
# s3ca.py backend adapter
# ---------------------------------------------------------------------------
class RFFASTBackend:
    """Adapter exposing R-FFAST as an `s3ca.py` backend: `required_indices`
    / `run`, matching `sfft_opt._SFFT1Backend` / `decimated_sfft._DecimatedBackend`.

    Like `_DecimatedBackend`, the front end is *fixed* at construction time
    (Section VII's whole point) -- `seed` is accepted by `required_indices`
    and `run` only for interface symmetry with the randomised `sfft1`
    backend, and is ignored. Build-time randomness (the cluster offsets
    r_i) is controlled by the constructor's own `seed`.
    """
    name = "rffast"

    def __init__(self, N, kappa, plan=None, d=3, f=None, clusters=None,
                 per_cluster=3, seed=0, iterations=8, noise_var=None,
                 gamma=0.5, **kw):
        if plan is None:
            plan = make_plan(N, kappa=kappa, f=f, d=d, clusters=clusters,
                              per_cluster=per_cluster, seed=seed)
        self.N, self.plan = N, plan
        self.beta = kay_weights(plan.shifts.shape[1])
        self.iterations = iterations
        self.noise_var = noise_var
        self.gamma = gamma
        self.kw = kw

    def required_indices(self, seed=None):
        return front_end_indices(self.plan)

    def run(self, x_col, kappa, seed=None):
        Ys = compute_front_end(x_col, self.plan)
        freqs, coeffs = peel_decode(
            Ys, self.plan, self.beta, kappa=kappa, iterations=self.iterations,
            noise_var=self.noise_var, gamma=self.gamma, **self.kw)
        return freqs, coeffs, {}


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def sparse_signal(n, k, snr_db=None):
        # SNR defined relative to *time-domain* signal power, matching the
        # convention already used by sfft_opt.py's and decimated_sfft.py's
        # own demos -- not relative to |X[l]|^2 directly. With numpy's FFT
        # convention (no 1/n baked into the forward transform), a length-n
        # signal built from O(1) spectral coefficients has tiny time-domain
        # samples (~1/n via Parseval), so scaling noise off |X[l]|^2 instead
        # would silently make "30 dB SNR" mean a noise power many times
        # *larger* than the actual time-domain signal -- an easy trap, and
        # the bug this self-test originally shipped with.
        X = np.zeros(n, dtype=complex)
        freqs = rng.choice(n, k, replace=False)
        v = rng.standard_normal(k) + 1j * rng.standard_normal(k)
        X[freqs] = v
        x = np.fft.ifft(X)                # so that np.fft.fft(x) == X exactly
        noise_var = 0.0
        if snr_db is not None:
            sig_pow = np.mean(np.abs(x) ** 2)
            noise_var = sig_pow / (10 ** (snr_db / 10))
            x = x + np.sqrt(noise_var / 2) * (rng.standard_normal(n)
                                               + 1j * rng.standard_normal(n))
        return x, X, freqs, noise_var

    print("R-FFAST self-test: exact recovery, no noise")
    n = 7 * 8 * 9          # = 504, three pairwise-coprime factors near k=8
    k = 8
    plan = make_plan(n, kappa=k, d=3, seed=0)
    print(f"  n={n}, f={plan.f.tolist()}, clusters x per_cluster = "
          f"{plan.shifts.shape}, samples/bin D={plan.shifts.size}")
    beta = kay_weights(plan.shifts.shape[1])
    idx = front_end_indices(plan)
    print(f"  required samples: {idx.size}/{n} ({100*idx.size/n:.1f}%)")

    for trial in range(5):
        x, X, freqs, _ = sparse_signal(n, k)
        Ys = compute_front_end(x, plan)
        fr, co = peel_decode(Ys, plan, beta, kappa=k, noise_var=1e-9)
        found = len(set(fr.tolist()) & set(freqs.tolist()))
        err = (np.max(np.abs(co - X[fr])) / np.max(np.abs(X))
               if fr.size else float("nan"))
        print(f"  trial {trial}: {found}/{k} found, rel err {err:.2e}")

    print("\nnoisy signals (n=504, k=8)")
    for snr in (30, 20, 10, 5):
        g = 0
        for _ in range(20):
            x, X, freqs, noise_var = sparse_signal(n, k, snr_db=snr)
            Ys = compute_front_end(x, plan)
            fr, co = peel_decode(Ys, plan, beta, kappa=k, noise_var=noise_var)
            g += len(set(fr.tolist()) & set(freqs.tolist()))
        print(f"  SNR {snr:>3} dB: {g/20:.2f}/{k} found on average")

    print("\nlarger case: n = 49*50*51 (paper's own example size), k = 40")
    n2 = 49 * 50 * 51
    plan2 = make_plan(n2, kappa=40, d=3, seed=1)
    beta2 = kay_weights(plan2.shifts.shape[1])
    idx2 = front_end_indices(plan2)
    print(f"  n={n2}, f={plan2.f.tolist()}, D={plan2.shifts.size}, "
          f"required samples: {idx2.size}/{n2} ({100*idx2.size/n2:.2f}%)")
    for snr in (20, 10, 5):
        g = 0
        for _ in range(10):
            x, X, freqs, noise_var = sparse_signal(n2, 40, snr_db=snr)
            Ys = compute_front_end(x, plan2)
            fr, co = peel_decode(Ys, plan2, beta2, kappa=40, noise_var=noise_var)
            g += len(set(fr.tolist()) & set(freqs.tolist()))
        print(f"  SNR {snr:>3} dB: {g/10:.2f}/40 found on average")

    print("\nchoose_stage_bins guardrail on a power-of-two n")
    try:
        choose_stage_bins(1 << 16, 100, d=3)
        print("  (unexpectedly succeeded)")
    except ValueError as e:
        print(f"  raised ValueError as expected: {e}")
