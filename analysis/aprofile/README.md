# Sparse-FFT engines for the Strip Spectral Correlation Analyzer (S3CA)

This package contains a full investigation into which sparse Fourier
transform (sparse-FFT) algorithm works best inside a Strip Spectral
Correlation Analyzer (S3CA) for estimating the spectral correlation density
(SCD) of cyclostationary signals at large input sizes.

## Start here

**`S3CA_sparse_FFT_investigation.ipynb`** — the comprehensive summary
notebook. Read this first. It contains the background, all five engines
compared head-to-head, the key results and figures, a tutorial on why
DSSS-BPSK is not sparse enough for exactly-sparse recovery methods, and the
full source code of every module written. Every number and figure in it was
produced by executing the code in-place.

## Headline finding

On the S3CA paper's own DSSS-BPSK test signal (kappa=50, Np=32, 10 dB SNR):

| backend | hit rate | notes |
|---|---|---|
| `sfft1` | 73.6% | randomized hashing; assumes *approximately* sparse — **best** |
| `decimated` | 23.3% | deterministic; high precision, low recall (abstains) |
| `rffast` | 5.3% | CRT multi-stage + peeling; hurt by aliasing loss × heavy tail |
| harmonic | 21.1% | recovers the fundamental alpha0 *exactly*; modest hit rate |
| debiasing | neutral | works on genuinely sparse signals; model-mismatch on BPSK |

Every method except `sfft1` assumes the cyclic spectrum is (near-)exactly
sparse. DSSS-BPSK's cyclic spectrum is only *approximately* sparse — its
top-50 coefficients hold just ~84% of a channel's energy, with the rest
spread across tens of thousands of bins. `sfft1` wins because it never
assumes the tail is zero. See the notebook's tutorial section for the full
explanation with worked examples.

## The other conclusion: change the question

If the actual goal is **detection/classification** (is a cyclic feature
present, and at what baud rate alpha0?) rather than reconstructing the whole
SCD surface, then the sparsity problem disappears entirely.
`cyclic_detect.py` estimates the cyclic autocorrelation directly at
candidate cycle frequencies — a parametric approach indifferent to the
broadband tail. On the same BPSK signal it recovers **alpha0 to <0.01%
error from ~3% of the samples**, detects reliably down to a few dB SNR, and
holds the false-alarm rate near its target. Head-to-head with the original
S3CA reconstruction: both recover alpha0 accurately (S3CA ~0.46%, parametric
~0.00–0.01%), but the parametric detector uses far fewer samples — while
S3CA gives you the whole surface. Match the tool to the mission.

`alpha_profile.py` extends this to the **complete alpha profile**
(P(alpha) = max_f |S_X^alpha(f)|, the paper's own Fig. 3 lower panels): the
profile is far sparser than the surface (~3% of alpha bins nonzero, all on
the harmonic comb), so it's recovered by estimating alpha0 and evaluating
only the comb points — landing on the dense profile at every populated
alpha from ~6% of the samples.

**On complexity — a real finding, not just intuition:** the naive way to
scan for alpha0 is compute-bound, not sample-bound — resolving a cyclic
peak needs grid resolution ~1/N, so a direct-summation scan costs
O(N) per lag despite reading only M<<N samples (worse than a dense
computation for M in the thousands; measured ~26s at N=131072, wide prior).
Fixed by recognizing this is an EXACT (not approximate) problem: subsampled
positions are already integers on the N-grid, so zero-padding into a
length-N buffer and taking one FFT gives the exact non-uniform DFT at every
native bin in O(N log N), independent of sample count. Measured speedup:
**~200-250x**, for identical accuracy — and the result is now faster than
*both* dense SSCA and S3CA's own sparse reconstruction at every N tested
(16K to 1M), because alpha0 estimation never needs to resolve spectral
frequency and so skips the Np-fold channelizer cost entirely. What it still
gives up: accuracy (S3CA's reconstruction is ~3x more accurate on the
profile, since it's a full-resolution transform vs. a lower-SNR estimate
from far fewer points) and generality (no full SCD surface).

## Files

### Original codebase (pre-existing)
- `s3ca.py` — the S3CA implementation with the pluggable backend protocol
  (`required_indices` / `run`), `dense_ssca` reference, and `check()` harness
- `sfft_opt.py` — optimized sFFT 1.0 backend (`sfft1`), with a
  Dolph-Chebyshev flat-top filter
- `sfft.py` — reference sFFT implementation
- `decimated_sfft.py` — deterministic decimation + binary-phase backend
- `demo_s3ca.py` — test-signal generators (`dsss_bpsk`, `am_tone`)
- `test_s3ca.py` — test suite

### Modules built during the investigation
- `rffast.py` — CRT-guided multi-stage aliasing sparse FFT + onion-peeling
  decoder (R-FFAST, Pawar & Ramchandran arXiv:1501.00320). Self-test:
  `python3 rffast.py`
- `fam.py` — FAM-style estimator using a smaller slow-time FFT, reusing the
  S3CA channelizer/CDP machinery
- `harmonic.py` — harmonic-structured recovery (fundamental search + joint
  comb least-squares refit)
- `debias.py` — backend-agnostic debiasing + iterative refinement wrapper
- `cyclic_detect.py` — parametric cyclic-feature detector/estimator: detects
  and estimates the fundamental cycle frequency alpha0 directly from
  subsampled data, bypassing SCD reconstruction. Self-test: `python3 cyclic_detect.py`
- `alpha_profile.py` — complete alpha-profile estimation: parametric alpha0
  + harmonic-comb evaluation. Recovers the full profile (every populated
  cycle frequency) from a few % of samples. Self-test: `python3 alpha_profile.py`
  Note: `estimate_alpha0`/`scan_alpha_full` in `cyclic_detect.py` default to
  the exact zero-padded-FFT scan (`fast=True`), ~200-250x faster than the
  original direct-summation grid search (`fast=False`, kept for comparison).
- `bpsk_compare.py` — the head-to-head comparison driver used throughout.
  Run directly: `python3 bpsk_compare.py --kappa 50`

### Notebooks and builders
- `S3CA_sparse_FFT_investigation.ipynb` — the main summary notebook (read this)
- `bpsk_comparison.ipynb` — the earlier, focused sfft1/decimated/rffast
  comparison notebook
- `build_summary_notebook.py` — regenerates the summary notebook (executes
  every cell, embeds real outputs/figures)
- `build_notebook.py` — regenerates `bpsk_comparison.ipynb`

## Running

All modules import against the pre-existing `s3ca.py` and its backend
protocol; no installation step beyond numpy + matplotlib. From this
directory:

```bash
python3 bpsk_compare.py --kappa 50        # head-to-head comparison
python3 rffast.py                          # R-FFAST self-test
python3 cyclic_detect.py                   # parametric detector self-test
python3 alpha_profile.py                   # alpha-profile estimator self-test
python3 build_summary_notebook.py          # regenerate the summary notebook
```

## Caveats worth carrying forward

- `rffast` needs the transform length N to factor into pairwise-coprime
  pieces near kappa; a power-of-two N cannot, so it runs on a nearby N.
- `rffast`/`fam` aliasing loss scales with P_s·N and f_s·P_s = N is a hard
  constraint — reducing the loss costs samples.
- `harmonic` needs a plausible baud-rate search range supplied; it breaks
  subharmonic ties toward the larger exact-multiple candidate.
- `debias` refinement needs its noise-floor guard on residual re-search
  (without it, spurious candidates crowd out real ones); it helps only when
  the signal is genuinely sparse.
- `cyclic_detect` needs a resolution-aware alpha grid (cyclic peaks are as
  sharp as full-resolution DFT bins) and a plausible alpha0 search range;
  its detection threshold is calibrated to the empirical H0 distribution,
  not a closed-form analytic one. Its scan cost is ~ n_alpha × n_samples,
  so it is cheapest on subsampled inputs (its intended regime).
