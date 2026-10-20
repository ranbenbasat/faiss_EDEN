# Matched fixed64 one/two-round EDEN quality control

Primary protocol preserves the archived native centroid-training procedure exactly. MSong420 pads to448; seven fixed64 groups, no permutation. Same first-round signs, independent second-round signs; seeds12345–12349, 4096training/20000base/200queries. Frozen native transform/codec/scanner, no timing. BF16 group scales plus globalFP32 original residual norm; same136/200/264B strides for both round counts.

|Bits|Recall@100 1round|Recall@100 2round|Pooled NMSE10k 1round|Pooled NMSE10k 2round|
|---|---:|---:|---:|---:|
|2|86.330%|87.022%|0.109879|0.123687|
|3|92.801%|92.875%|0.0297825|0.0320277|
|4|96.081%|95.961%|0.00775884|0.00868646|

Two rounds worsen pooled NMSE and mean per-vector NMSE at all widths. Recall is not monotonic with reconstruction error: mean Recall@10/100 increases at2/3bits, decreases at4bits. This is a fixed-data five-seed pilot, not a general claim that either round count always wins. Summary includes paired seed deltas and conditional seed-bootstrap intervals.

FP32 group-scale diagnostic preserves identical code assignments to BF16 within each round count and checks actual BF16 scales are RNE-rounded FP32 scales. The one/two-round pattern persists: BF16 rounding is not the main explanation. FP32 uses144/200/264B; 2bit is over the136B budget and is diagnostic only.

All30 two-round BF16/FP32 cases exactly reproduce archived packed code prefixes and per-query Recall@10/100; NMSE/vNMSE reproduce within1e-14 relative. Four-bit FP32 has8 trailing unused bytes versus its archived256B compact record. Both round counts pass independent explicit H64 parity-matrix checks on the full448 basis for all5seeds, exact support locality, shared first signs, and block energy preservation checks on data.

A separate60-case common-original-centroid diagnostic is labelled `fixed64-rounds-matched-*`. It transforms a single original-space native centroid instead of independently accumulating post-rotation means. It yields the same substantive pattern; it is NOT substituted for the exact archived-centering primary.

Primary raw/summary: `fixed64-rounds-archivedcenter-quality.json` / `fixed64-rounds-archivedcenter-summary.json`. Includes Recall@10/100, pooled NMSE/vNMSE first10k and full20k, all60 unique settings, actual roundcounts/sign hashes, native binary hash, per-group error/energy summary. Native DLL remains875e2e701f56b049b828017e1d2092e77fb15b5b093694b17d504bb5b7209ab0.

Large paired NPZ detail is retained locally and separately hashed in manifests; it is NOT embedded in the compact public capsule. It contains per-vector error/norm, actual stored scales/globalnorm/signs, per-query recall/predictions/distances; the primary adds per-group error/energy. No source embeddings or binaries are in the compact capsule. No new speed measurements or OS changes were made in these quality runs.
