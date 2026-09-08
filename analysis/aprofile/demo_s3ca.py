"""Demo / check, mirroring the paper's Sec. IV-A test signal:
a DSSS-BPSK signal, 10 dB SNR, processing gain 31, chip rate 0.25,
sample rate normalised to 1 -- cycle frequencies at multiples of the
data rate (0.25/31). Compares S3CA built on the "sfft1" backend
(sfft_opt.py, random hashing) against the "decimated" backend
(decimated_sfft.py, fixed decimation + binary phase encoding)."""
import numpy as np

from s3ca import compare_backends, print_comparison


def dsss_bpsk(n_symbols, chip_rate=0.25, gain=31, snr_db=10.0, rng=None):
    rng = np.random.default_rng(rng)
    sps = round(1.0 / chip_rate)                    # samples per chip
    code = rng.choice([-1.0, 1.0], size=gain)         # fixed spreading code
    symbols = rng.choice([-1.0, 1.0], size=n_symbols)  # data bits
    chips = symbols[:, None] * code[None, :]           # (n_symbols, gain)
    chips = chips.ravel()
    x = np.repeat(chips, sps).astype(complex)          # rectangular pulse shaping

    sig_power = np.mean(np.abs(x) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    x = x + noise

    data_rate = chip_rate / gain
    return x, data_rate


def am_tone(N, alpha0, f0=0.05, snr_db=30.0, rng=None):
    """A genuinely cyclic-frequency-sparse signal: an amplitude-modulated
    tone. Its SCD, at spectral frequency f0, has support at only a few cycle
    frequencies (0, +-alpha0, +-2*alpha0, ...) -- the regime `decimated_sfft`
    (a singleton/collision detector) is built for, unlike a real comm
    signal's CDP slice, which is closer to continuously spread."""
    rng = np.random.default_rng(rng)
    n = np.arange(N)
    x = (1.0 + 0.6 * np.cos(2 * np.pi * alpha0 * n + rng.uniform(0, 2 * np.pi))) \
        * np.exp(2j * np.pi * f0 * n)
    sig_power = np.mean(np.abs(x) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    x = x + np.sqrt(noise_power / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    return x


if __name__ == "__main__":
    chip_rate, gain = 0.25, 31
    sps = round(1.0 / chip_rate)
    data_rate = chip_rate / gain

    print("#" * 78)
    print("# Scenario 1: DSSS-BPSK (paper's own test signal) -- CDP columns are NOT")
    print("# strongly cyclic-frequency-sparse; see fill_rate below.")
    print("#" * 78)
    for lg, Np in ((16, 32), (18, 64), (20, 16)):
        N = 1 << lg
        n_symbols = N // (sps * gain) + 1
        x, dr = dsss_bpsk(n_symbols, chip_rate=chip_rate, gain=gain, rng=lg)
        x = x[:N]
        kappa = 50

        print(f"\n=== N = 2^{lg} ===")
        reports = compare_backends(
            x, Np, kappa, seed=3, expected_data_rate=dr, alpha_tol_bins=2,
            backend_kwargs={
                "sfft1": {"loc_loops": 4, "est_loops": 8, "tolerance": 1e-4},
                "decimated": {"window": "kaiser", "beta": 12.0},
            })
        print_comparison(reports)

    print()
    print("#" * 78)
    print("# Scenario 2: AM tone -- genuinely cyclic-frequency-sparse per channel")
    print("# (decimated_sfft's home turf: a clean singleton-test regime)")
    print("#" * 78)
    N, Np, kappa = 1 << 18, 32, 8
    alpha0 = 37.0 / N          # an off-grid-ish, but well-separated, cycle frequency
    x = am_tone(N, alpha0, f0=0.05, snr_db=30.0, rng=1)
    from s3ca import check
    reports = {
        "sfft1": check(x, Np, kappa, backend="sfft1", seed=3,
                        loc_loops=4, est_loops=8, tolerance=1e-4),
        "decimated (default)": check(x, Np, kappa, backend="decimated", D=16,
                                      window="kaiser", beta=12.0),
        "decimated (tuned)": check(x, Np, kappa, backend="decimated", D=16,
                                    window="kaiser", beta=12.0, noise_factor=3.0),
    }
    print_comparison(reports)
    print("\n(the 'tuned' row lowers decode()'s noise_factor from its default 6.0 to 3.0:")
    print(" fill rate jumps a lot -- it stops abstaining -- but hit rate barely moves,")
    print(" because most of what it now reports isn't among the dense top-kappa either.")
    print(" Loosening the threshold buys quantity, not accuracy, on this signal.)")



