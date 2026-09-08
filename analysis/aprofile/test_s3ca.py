"""Automated checks for s3ca.py.

Run with:  python3 test_s3ca.py
Exits non-zero (and prints which check failed) if anything is wrong.
"""
from math import gcd

import numpy as np

from sfft_opt import flat_filter, sfft1
from decimated_sfft import make_plan
from s3ca import (dense_ssca, required_time_indices, s3ca, check, compare_backends,
                   make_backend, _channelizer_footprint, _DecimatedBackend)


def _brute_dense_ssca(x, Np, a, g):
    """O(N^2 * Np) reference with no vectorization at all -- ground truth
    for validating `dense_ssca`'s formulas independently."""
    N = x.size
    k_idx = np.arange(-Np // 2, Np - Np // 2)
    q_idx = np.arange(-N // 2, N - N // 2)
    XT = np.zeros((N, Np), dtype=complex)
    for t in range(N):
        for ki, k in enumerate(k_idx):
            fk = k / Np
            s = 0j
            for ri, r in enumerate(range(-Np // 2, Np - Np // 2)):
                s += a[ri] * x[(t + r) % N] * np.exp(-2j * np.pi * fk * r)
            XT[t, ki] = s * np.exp(-2j * np.pi * fk * t)
    Xg = XT * np.conj(x)[:, None] * g[:, None]
    S = np.zeros((N, Np), dtype=complex)
    for qi, q in enumerate(q_idx):
        for ki in range(Np):
            S[qi, ki] = np.sum(Xg[:, ki] * np.exp(-2j * np.pi * q * np.arange(N) / N))
    return S


def test_dense_ssca_matches_brute_force():
    rng = np.random.default_rng(1)
    N, Np = 32, 8
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    a, g = np.hamming(Np), np.hamming(N)
    S_brute = _brute_dense_ssca(x, Np, a, g)
    S, f, alpha = dense_ssca(x, Np, a=a, g=g)
    err = np.max(np.abs(S_brute - S))
    assert err < 1e-9, f"dense_ssca disagrees with brute force, max err {err}"

    # spot-check the (f, alpha) coordinate map against Eq. (3) directly
    k_idx = np.arange(-Np // 2, Np - Np // 2)
    q_idx = np.arange(-N // 2, N - N // 2)
    for qi, ki in [(3, 2), (0, 0), (N - 1, Np - 1)]:
        k, q = k_idx[ki], q_idx[qi]
        fk, dalpha = k / Np, 1.0 / N
        assert np.isclose(alpha[qi, ki], fk + q * dalpha)
        assert np.isclose(f[qi, ki], (fk - q * dalpha) / 2)


def test_required_indices_match_sfft1_internal_rng():
    """required_time_indices must reproduce sfft1's own sigma/tau draw
    exactly, or the whole shared-seed trick is unfounded."""
    N, B, loc_loops, est_loops, seed = 4096, 256, 4, 8, 7
    filt = flat_filter(N, B)
    _, sigmas, taus = required_time_indices(N, filt, loc_loops, est_loops, seed)

    rng = np.random.default_rng(seed)
    L = loc_loops + est_loops
    sig2 = np.empty(L, dtype=np.int64)
    tau2 = np.empty(L, dtype=np.int64)
    for i in range(L):
        s = int(rng.integers(0, N))
        while gcd(s, N) != 1:
            s = int(rng.integers(0, N))
        sig2[i], tau2[i] = s, int(rng.integers(0, N))
    assert np.array_equal(sigmas, sig2), "sigma sequence diverged from sfft1's internal draw"
    assert np.array_equal(taus, tau2), "tau sequence diverged from sfft1's internal draw"


def test_masking_outside_wprime_never_changes_sfft1_output():
    """The load-bearing claim: if sfft1 is only ever going to read indices in
    W', then a signal that agrees with the true one on W' and is arbitrary
    (here: zero) elsewhere must give byte-identical output, for the same
    seed. This is what licenses computing the CDP only on W'."""
    N, kappa, B, loc_loops, est_loops, seed = 4096, 10, 256, 4, 8, 7
    filt = flat_filter(N, B)
    Wp_idx, _, _ = required_time_indices(N, filt, loc_loops, est_loops, seed)

    rng = np.random.default_rng(0)
    Xk = np.zeros(N, dtype=complex)
    Xk[:5] = (rng.standard_normal(5) + 1j * rng.standard_normal(5)) * 20
    x_full = (np.fft.ifft(Xk) * N).astype(complex)

    masked = np.zeros(N, dtype=complex)
    masked[Wp_idx] = x_full[Wp_idx]

    f1, c1 = sfft1(x_full, kappa, filt=filt, loc_loops=loc_loops, est_loops=est_loops, rng=seed)
    f2, c2 = sfft1(masked, kappa, filt=filt, loc_loops=loc_loops, est_loops=est_loops, rng=seed)
    assert np.array_equal(f1, f2) and np.allclose(c1, c2), \
        "sfft1 touched something outside W' -- the shared-seed restriction is unsound"


def test_full_mode_never_reads_outside_wprime_in_practice():
    """End-to-end version of the previous check, through s3ca() itself:
    compare s3ca(mode='full') against a version where we deliberately corrupt
    every entry of x OUTSIDE the channelizer's true footprint (W' dilated by
    the Np-sample channelizer window -- see `_channelizer_footprint`) with
    garbage. If 'full' mode is truly only touching that footprint, the
    answer must not change."""
    rng = np.random.default_rng(2)
    N, Np, kappa = 8192, 8, 10
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    B = 256
    filt = flat_filter(N, B)
    seed = 5

    r1 = s3ca(x, Np, kappa, mode="full", seed=seed, filt=filt, loc_loops=4, est_loops=8)

    Wp_idx, _, _ = required_time_indices(N, filt, 4, 8, seed)
    footprint = _channelizer_footprint(Wp_idx, N, Np)
    x_corrupt = x.copy()
    mask = np.ones(N, dtype=bool)
    mask[footprint] = False
    x_corrupt[mask] = rng.standard_normal(mask.sum()) + 1j * rng.standard_normal(mask.sum())
    r2 = s3ca(x_corrupt, Np, kappa, mode="full", seed=seed, filt=filt, loc_loops=4, est_loops=8)

    assert np.array_equal(r1.channel, r2.channel)
    assert np.allclose(r1.value, r2.value)
    assert np.allclose(r1.alpha, r2.alpha)


def test_check_runs_and_reports_sane_numbers():
    rng = np.random.default_rng(3)
    N, Np, kappa = 1 << 14, 16, 20
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    x[::37] += 5.0  # inject some structure so it isn't pure noise
    r = check(x, Np, kappa, seed=1, loc_loops=4, est_loops=8, tolerance=1e-4)
    assert 0.0 <= r.full_hit_rate <= 1.0
    assert 0.0 <= r.naive_hit_rate <= 1.0
    assert 0.0 < r.full_sparsity_ratio <= 1.0
    assert r.full_time > 0 and r.naive_time > 0 and r.dense_time > 0


def test_decimated_backend_required_indices_are_seed_independent():
    """decimated_sfft has no randomness: the required-sample set must not
    depend on `seed` at all, unlike the sfft1 backend."""
    N, kappa = 4096, 10
    be = _DecimatedBackend(N, kappa, D=16, window="rect")
    idx_a = be.required_indices(seed=0)
    idx_b = be.required_indices(seed=12345)
    assert np.array_equal(np.sort(idx_a), np.sort(idx_b))
    # and it should match the paper's own stated duty cycle (log2(D)+1)/D
    plan = be.plan
    expected_size = (int(np.log2(plan.D)) + 1) * plan.B
    assert idx_a.size == expected_size


def test_decimated_backend_recovers_exact_tones_or_flags_collisions():
    """On-grid tones with no bin collision must be recovered exactly; any
    miss must be explained by a genuine collision (both frequencies sharing
    a bin mod B), which the algorithm should flag in `unresolved` rather
    than silently drop or guess at."""
    rng = np.random.default_rng(4)
    N, kappa = 1 << 14, 8
    Xk = np.zeros(N, dtype=complex)
    freqs_true = rng.choice(N, kappa, replace=False)
    Xk[freqs_true] = rng.standard_normal(kappa) + 1j * rng.standard_normal(kappa)
    x = np.fft.ifft(Xk) * N

    be = make_backend(N, kappa, backend="decimated", D=16, window="rect")
    plan = be.plan
    freqs, coeffs, extra = be.run(x, kappa, seed=None)

    # recompute `unresolved` directly to check the miss/collision correspondence
    buf = np.empty((len(plan.taus), plan.B), dtype=complex)
    for bi, tau in enumerate(plan.taus):
        buf[bi] = x[int(tau)::plan.D]
    Y = plan.scale * np.fft.fft(buf * plan.window, axis=-1)
    from decimated_sfft import decode
    _, _, unresolved = decode(Y, plan)

    missing = set(freqs_true.tolist()) - set(freqs.tolist())
    for m in missing:
        assert (m % plan.B) in unresolved.tolist(), \
            f"frequency {m} went missing without a flagged collision"
    assert len(missing) < kappa, "found nothing at all -- something is broken"


def test_full_mode_never_reads_outside_wprime_decimated():
    """Same load-bearing claim as the sfft1 version, for the decimated
    backend: corrupting everything outside its (deterministic) required set
    must not change s3ca(mode='full')'s output."""
    rng = np.random.default_rng(5)
    N, Np, kappa = 8192, 8, 10
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    be = make_backend(N, kappa, backend="decimated", D=32, window="rect")

    r1 = s3ca(x, Np, kappa, mode="full", backend=be)

    Wp_idx = be.required_indices(None)
    footprint = _channelizer_footprint(Wp_idx, N, Np)
    x_corrupt = x.copy()
    mask = np.ones(N, dtype=bool)
    mask[footprint] = False
    x_corrupt[mask] = rng.standard_normal(mask.sum()) + 1j * rng.standard_normal(mask.sum())
    r2 = s3ca(x_corrupt, Np, kappa, mode="full", backend=be)

    assert np.array_equal(r1.channel, r2.channel)
    assert np.allclose(r1.value, r2.value)


def test_naive_and_full_agree_for_deterministic_backend():
    """Since decimated_sfft's required set doesn't depend on seed or
    channel, naive and full S3CA should recover the same thing (up to the
    dense-vs-restricted channelizer being mathematically identical on the
    samples that matter)."""
    rng = np.random.default_rng(6)
    N, Np, kappa = 8192, 8, 10
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    be = make_backend(N, kappa, backend="decimated", D=32, window="rect")

    naive = s3ca(x, Np, kappa, mode="naive", backend=be)
    full = s3ca(x, Np, kappa, mode="full", backend=be)
    assert naive.n_positions == N
    assert full.n_positions < naive.n_positions
    order_n = np.lexsort((naive.f, naive.channel))
    order_f = np.lexsort((full.f, full.channel))
    assert np.allclose(naive.value[order_n], full.value[order_f], atol=1e-8)


def test_compare_backends_runs_both():
    rng = np.random.default_rng(7)
    N, Np, kappa = 1 << 14, 16, 20
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    x[::29] += 4.0
    reports = compare_backends(
        x, Np, kappa,
        backend_kwargs={"sfft1": {"loc_loops": 4, "est_loops": 8, "tolerance": 1e-4},
                         "decimated": {"D": 64, "window": "rect"}})
    assert set(reports) == {"sfft1", "decimated"}
    for name, r in reports.items():
        assert r.backend == name
        assert 0.0 <= r.full_hit_rate <= 1.0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    if failed:
        raise SystemExit(f"\n{failed}/{len(tests)} checks failed")
    print(f"\nall {len(tests)} checks passed")
