# Initial Amumu MSong pilot — 20 October 2026

This is a completed flat compressed-scan pilot: 150 cases covering RaBitQ and EDEN, 2/3/4 stored bits, none/Haar URR/1/2/3 RHTs, and five paired transform seeds. The source was compiled and validated on Amumu for its Arrow Lake CPU before measurement. Runtime figures below remain provisional because desktop activity exceeded the declared threshold in 14 of 95 workload samples.

Training/base/query sizes are 4,096/20,000/200. Training uses source base rows [0,4096), searched base rows [4096,24096), and query rows [0,200). These are disjoint training/base source-row ranges from a prefix subset. The exact-value check found **21 searched base rows whose float32 vectors also occur in training**; this is not a value-disjoint split. Randomized and full-dataset validation remain pending. Exact original-space L2 neighbors were recomputed for this searched subset. Recall is set Recall@100, with query batch 256, one warmup and three timed repetitions, one pinned performance core and one verified BLAS/OpenMP thread.

## Recall and paired uncertainty

All RHT depths have the same 512-coordinate output. Deltas are percentage points versus 1RHT; brackets show paired two-way seed/query percentile 95% bootstrap intervals (1,000 draws). They condition on this fixed training/base/query population and five transform seeds.

| Quantizer | Bits | 1RHT recall % | 2RHT recall % | 3RHT recall % | 2RHT delta pp [95% CI] | 3RHT delta pp [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| RaBitQ | 2 | 86.657 | 88.891 | 88.919 | 2.234 [1.616, 2.901] | 2.262 [1.639, 2.943] |
| RaBitQ | 3 | 94.039 | 93.849 | 93.798 | -0.190 [-0.463, 0.081] | -0.241 [-0.460, -0.034] |
| RaBitQ | 4 | 96.478 | 96.512 | 96.464 | 0.034 [-0.113, 0.178] | -0.014 [-0.185, 0.165] |
| EDEN | 2 | 88.682 | 88.927 | 88.997 | 0.245 [-0.170, 0.644] | 0.315 [-0.130, 0.729] |
| EDEN | 3 | 94.062 | 93.947 | 94.013 | -0.115 [-0.394, 0.159] | -0.049 [-0.288, 0.173] |
| EDEN | 4 | 96.857 | 96.775 | 96.755 | -0.082 [-0.262, 0.113] | -0.102 [-0.279, 0.062] |

The 2-bit RaBitQ improvement has a positive paired interval for both added depths. EDEN intervals include zero at every tested bit width. At 3-bit RaBitQ, 3RHT has a small negative interval. These outcomes motivate broader validation of a selectable depth; they do not establish a universal preferred depth.

## Reconstruction and provisional full-search speed

Pooled residual NMSE and mean per-vector residual vNMSE are measured on the first 10,000 searched vectors in transformed coordinates. They include padded off-subspace error. These use the actual native reconstruction scale, which differs from the paper's best-scalar residual reconstruction diagnostic. Full-search QPS includes query preprocessing and Python batching. Its qualification is provisional for every row in this pilot; no speedup claim is established here.

| Quantizer | Bits | Mode | Recall@100 % | NMSE | vNMSE | Provisional QPS | Output dimension | Code bytes/vector |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| RaBitQ | 2 | none | 38.720 | 7.109675 | 1.480424 | 1576.1 | 420 | 126 |
| RaBitQ | 2 | urr | 87.753 | 0.145376 | 0.140738 | 1868.2 | 420 | 126 |
| RaBitQ | 2 | rht1 | 86.657 | 0.094665 | 0.124497 | 2800.0 | 512 | 148 |
| RaBitQ | 2 | rht2 | 88.891 | 0.138302 | 0.138385 | 2950.0 | 512 | 148 |
| RaBitQ | 2 | rht3 | 88.919 | 0.140232 | 0.140155 | 2932.4 | 512 | 148 |
| RaBitQ | 3 | none | 56.025 | 2.136703 | 0.571724 | 1448.4 | 420 | 178 |
| RaBitQ | 3 | urr | 93.064 | 0.040783 | 0.038914 | 1763.7 | 420 | 178 |
| RaBitQ | 3 | rht1 | 94.039 | 0.023286 | 0.032430 | 2400.5 | 512 | 212 |
| RaBitQ | 3 | rht2 | 93.849 | 0.038194 | 0.037861 | 2589.7 | 512 | 212 |
| RaBitQ | 3 | rht3 | 93.798 | 0.038411 | 0.038610 | 2567.6 | 512 | 212 |
| RaBitQ | 4 | none | 69.155 | 0.615978 | 0.200643 | 1329.0 | 420 | 231 |
| RaBitQ | 4 | urr | 96.014 | 0.013610 | 0.012659 | 1644.2 | 420 | 231 |
| RaBitQ | 4 | rht1 | 96.478 | 0.008474 | 0.010452 | 2135.7 | 512 | 276 |
| RaBitQ | 4 | rht2 | 96.512 | 0.012142 | 0.012068 | 2280.2 | 512 | 276 |
| RaBitQ | 4 | rht3 | 96.464 | 0.012168 | 0.012298 | 2269.0 | 512 | 276 |
| EDEN | 2 | none | 39.635 | 5.816595 | 1.295137 | 1225.1 | 420 | 113 |
| EDEN | 2 | urr | 87.785 | 0.138461 | 0.133848 | 1239.2 | 420 | 113 |
| EDEN | 2 | rht1 | 88.682 | 0.110284 | 0.119164 | 1038.4 | 512 | 136 |
| EDEN | 2 | rht2 | 88.927 | 0.132201 | 0.131454 | 1045.7 | 512 | 136 |
| EDEN | 2 | rht3 | 88.997 | 0.133358 | 0.133406 | 1032.9 | 512 | 136 |
| EDEN | 3 | none | 49.020 | 1.613527 | 0.555307 | 205.6 | 420 | 166 |
| EDEN | 3 | urr | 93.294 | 0.037742 | 0.036222 | 205.3 | 420 | 166 |
| EDEN | 3 | rht1 | 94.062 | 0.031162 | 0.032795 | 169.5 | 512 | 200 |
| EDEN | 3 | rht2 | 93.947 | 0.035449 | 0.035273 | 169.6 | 512 | 200 |
| EDEN | 3 | rht3 | 94.013 | 0.035523 | 0.035821 | 169.1 | 512 | 200 |
| EDEN | 4 | none | 54.705 | 0.711501 | 0.351242 | 639.8 | 420 | 218 |
| EDEN | 4 | urr | 96.373 | 0.010552 | 0.009840 | 646.7 | 420 | 218 |
| EDEN | 4 | rht1 | 96.857 | 0.008374 | 0.008851 | 528.7 | 512 | 264 |
| EDEN | 4 | rht2 | 96.775 | 0.009414 | 0.009401 | 530.4 | 512 | 264 |
| EDEN | 4 | rht3 | 96.755 | 0.009422 | 0.009581 | 532.9 | 512 | 264 |

For both quantizers, 2/3RHT pooled NMSE and vNMSE are worse than 1RHT at all three tested budgets. This illustrates why reconstruction error and nearest-neighbor recall must be reported separately. Native-size none/URR retain 420 coordinates, while RHT outputs 512; comparisons against those two modes require the queued matched-padding control and physical storage figures above.

## Provenance and remaining evidence

- Host: Amumu, Intel Core Ultra 9 285K; eight performance and sixteen efficiency cores, about 127 GiB RAM. Measurement affinity: logical CPU 0, a verified performance core.
- Frozen local source: `159b1c54a36aca8dad7c369661b15d86b27e516f`, derived from PR source `50ae699ad899ed6f2eb38794a8f34ff6dd2d22b1` plus the archived MinGW portability patch.
- Build: GCC 16.2, OpenBLAS 0.3.34, OpenMP 5.2, Python 3.14; Release `-O3 -march=native -ffp-contract=off`, AVX2 target. Compiler vectorization diagnostics and 28 passing focused tests are archived.
- Threadpoolctl 3.6 provides verified OpenBLAS/OpenMP thread control in this isolated Windows environment.
- Raw JSON SHA-256: `31fe5946c3c4f650542d3c8c9e27a243b63dd3bfe14d5231cfaccd62e54e8132`.
- Compressed evidence capsule SHA-256: `f4fabcc3380726803d5cba06539de412010e954d6f4f46ad47614db51f787cfe`.

[Machine-readable paired comparisons](first-pilot-summary.json), [capsule transfer manifest](transfer-manifest.json), [exact portability patch](mingw-portability.patch), and [base64 tar.xz evidence capsule](evidence.tar.xz.base64) are retained on this evidence branch. The capsule contains the full raw report, split/build/dependency manifests, tuning/correctness logs and ANN driver source. The coordinator downloaded, decoded and independently verified both archive and raw-result hashes.

The native-dimension IVF/EDEN, IVF/RaBitQ and IVF/RaBitQFastScan nprobe sweeps have completed separately on the same build; matched-padding sweeps are running. Source audit confirms that this flat pilot differs from the historical MSong figure in transform construction, database quantizer scaling, residual centers, query precision, data population and reconstruction metric. Controlled quality ablations are queued on Amumu before interpreting that discrepancy. GIST/Word2Vec, one-bit, randomized-split/larger-data validation and quieter timing repeats remain required evidence. Flat results above are labeled independently of ANN results.
