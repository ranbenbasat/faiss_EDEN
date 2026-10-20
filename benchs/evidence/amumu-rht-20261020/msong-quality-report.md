# Matched MSong quality diagnostic on Amumu

The completed bounded diagnostic supports the transform construction as a major difference between the historical probe and the native pilot. With the same data, centroids and historical scalar thresholds, the older 448-dimensional block/Kac transform has a weaker one-round baseline and large depth gains. The full 512-dimensional transform starts higher and gains little at these settings.

This comparison changes both transform construction and coded dimension: 512 has 14.3% more code coordinates than 448. It does NOT isolate the pairwise Kac layer, and does not prove that adding that layer improves a full RHT. A same-dimension, with/without-layer control has been requested on Amumu.

## Closest bounded match to the paper's clustered fixed-threshold setup

All rows below use the same 128 frozen centroids and historical fixed thresholds. Recall@100 percentages average two transform seeds on 20,000 searched vectors and 200 queries. Centroids were fitted on the pilot training set, not the full historical database. These are diagnostic oracles, not native IVF search timings.

| Bits | Block/Kac 448: 1 / 2 / 3 rounds (%) | Full 512: 1 / 2 / 3 rounds (%) |
|---:|---|---|
| 2 | 89.7300 / 93.1600 / 93.4775 | 93.8900 / 94.0450 / 93.9825 |
| 3 | 94.4875 / 95.8075 / 96.0875 | 96.1875 / 96.1100 / 96.1475 |
| 4 | 95.9325 / 97.6475 / 97.9750 | 98.1100 / 98.0375 / 98.0850 |

Full-512 one-round recall exceeds older block/Kac three-round recall at every bit width in this stratum. That describes absolute quality on this population; it is not a universal ordering or an attribution to Kac alone.

## What the other controls establish

- Correcting the historical calibration search does not eliminate the large depth gains within the older construction. The calibration defect alone does not explain that pattern.
- Native per-residual code scaling also retains larger depth gains in the older construction than in full-512 for the 128-center control. This difference therefore persists across the three tested scalar-code choices.
- Query precision is not a material explanation in the native flat pilot: qb0 versus qb4 changes mean recall by at most 0.002 percentage points. All 45 qb4 replays exactly match the original pilot's per-query recalls, and query settings use identical stored database codes.
- Both best-scalar vNMSE and actual-score reconstruction errors are retained. For full-512, an extra round need not reduce either reconstruction statistic, even when recall improves.

## Qualifications

The legacy geometry follows frozen source, but its imported historical FWHT dependency and Torch/GPU RNG are not completely archived. The diagnostic uses two NumPy sign seeds and normalized float64 transform oracles. It covers a 20k prefix and 200 queries rather than 994,185/1,000; the 128 centroids are native Lloyd fits to 4,096 training vectors. The fixed 448-dimensional threshold constants were deliberately reused at 512 without recalibration. Native per-residual factors include native code and float32 conventions. All outcomes and source are archived; no speed claim comes from this diagnostic.

## All diagnostic recall curves

| Centers | Scalar code | Transform | Bits | Recall %: 1 / 2 / 3 | Delta pp: 2 / 3 vs 1 |
|---:|---|---|---:|---|---|
| 1 | historical_fixed_t | legacy448_description | 2 | 83.6450 / 87.5700 / 88.0750 | +3.9250 / +4.4300 |
| 1 | historical_fixed_t | legacy448_description | 3 | 90.6125 / 92.1750 / 92.1625 | +1.5625 / +1.5500 |
| 1 | historical_fixed_t | legacy448_description | 4 | 94.2825 / 95.8350 / 95.9850 | +1.5525 / +1.7025 |
| 1 | historical_fixed_t | full512_oracle | 2 | 88.4325 / 88.6725 / 88.7925 | +0.2400 / +0.3600 |
| 1 | historical_fixed_t | full512_oracle | 3 | 92.2650 / 92.3075 / 92.4000 | +0.0425 / +0.1350 |
| 1 | historical_fixed_t | full512_oracle | 4 | 96.2975 / 96.2425 / 96.1000 | -0.0550 / -0.1975 |
| 1 | corrected_fixed_t | legacy448_description | 2 | 83.6450 / 87.5700 / 88.0750 | +3.9250 / +4.4300 |
| 1 | corrected_fixed_t | legacy448_description | 3 | 90.0800 / 92.8850 / 93.4075 | +2.8050 / +3.3275 |
| 1 | corrected_fixed_t | legacy448_description | 4 | 93.8300 / 95.8775 / 96.0850 | +2.0475 / +2.2550 |
| 1 | corrected_fixed_t | full512_oracle | 2 | 88.4325 / 88.6725 / 88.7925 | +0.2400 / +0.3600 |
| 1 | corrected_fixed_t | full512_oracle | 3 | 93.6000 / 93.7325 / 93.5725 | +0.1325 / -0.0275 |
| 1 | corrected_fixed_t | full512_oracle | 4 | 96.4300 / 96.3600 / 96.3225 | -0.0700 / -0.1075 |
| 1 | native_per_residual | legacy448_description | 2 | 83.7300 / 87.4650 / 88.1550 | +3.7350 / +4.4250 |
| 1 | native_per_residual | legacy448_description | 3 | 90.9225 / 92.8200 / 93.3625 | +1.8975 / +2.4400 |
| 1 | native_per_residual | legacy448_description | 4 | 94.6375 / 95.9125 / 96.1675 | +1.2750 / +1.5300 |
| 1 | native_per_residual | full512_oracle | 2 | 86.8975 / 89.0100 / 88.8600 | +2.1125 / +1.9625 |
| 1 | native_per_residual | full512_oracle | 3 | 94.0425 / 93.8600 / 93.8150 | -0.1825 / -0.2275 |
| 1 | native_per_residual | full512_oracle | 4 | 96.6175 / 96.5275 / 96.4825 | -0.0900 / -0.1350 |
| 128 | historical_fixed_t | legacy448_description | 2 | 89.7300 / 93.1600 / 93.4775 | +3.4300 / +3.7475 |
| 128 | historical_fixed_t | legacy448_description | 3 | 94.4875 / 95.8075 / 96.0875 | +1.3200 / +1.6000 |
| 128 | historical_fixed_t | legacy448_description | 4 | 95.9325 / 97.6475 / 97.9750 | +1.7150 / +2.0425 |
| 128 | historical_fixed_t | full512_oracle | 2 | 93.8900 / 94.0450 / 93.9825 | +0.1550 / +0.0925 |
| 128 | historical_fixed_t | full512_oracle | 3 | 96.1875 / 96.1100 / 96.1475 | -0.0775 / -0.0400 |
| 128 | historical_fixed_t | full512_oracle | 4 | 98.1100 / 98.0375 / 98.0850 | -0.0725 / -0.0250 |
| 128 | corrected_fixed_t | legacy448_description | 2 | 89.7300 / 93.1600 / 93.4775 | +3.4300 / +3.7475 |
| 128 | corrected_fixed_t | legacy448_description | 3 | 93.5200 / 96.0425 / 96.5000 | +2.5225 / +2.9800 |
| 128 | corrected_fixed_t | legacy448_description | 4 | 95.4825 / 97.5700 / 97.9450 | +2.0875 / +2.4625 |
| 128 | corrected_fixed_t | full512_oracle | 2 | 93.8900 / 94.0450 / 93.9825 | +0.1550 / +0.0925 |
| 128 | corrected_fixed_t | full512_oracle | 3 | 96.6325 / 96.6725 / 96.6950 | +0.0400 / +0.0625 |
| 128 | corrected_fixed_t | full512_oracle | 4 | 98.1275 / 98.1500 / 98.2025 | +0.0225 / +0.0750 |
| 128 | native_per_residual | legacy448_description | 2 | 89.7050 / 92.9025 / 93.4775 | +3.1975 / +3.7725 |
| 128 | native_per_residual | legacy448_description | 3 | 94.4850 / 96.0575 / 96.5625 | +1.5725 / +2.0775 |
| 128 | native_per_residual | legacy448_description | 4 | 96.7725 / 97.6025 / 97.9075 | +0.8300 / +1.1350 |
| 128 | native_per_residual | full512_oracle | 2 | 93.5300 / 93.8850 / 93.9450 | +0.3550 / +0.4150 |
| 128 | native_per_residual | full512_oracle | 3 | 96.6250 / 96.6950 / 96.7325 | +0.0700 / +0.1075 |
| 128 | native_per_residual | full512_oracle | 4 | 98.0675 / 98.0850 / 98.0875 | +0.0175 / +0.0200 |

Verified raw SHA-256: `9cd57245e2ed2d6838617ae642ed4967e5f2912dfbcbadc56e60daf6469e6f48`. Full coverage: 216 unique factor cases, 90 native query-bit rows, 200 per-query recalls per row. Transfer manifest and source capsule are adjacent to this report.
