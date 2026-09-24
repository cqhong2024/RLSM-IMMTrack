# Archived experiment results

These are local re-evaluation records from the manuscript experiments, not official leaderboard scores. The release process re-scored all available archived prediction boxes with the packaged metric functions; it did not rerun full neural inference on every sequence.

- benchmark_metrics.csv: pooled full-precision metrics, evaluated/absent counts, publication year and historical FPS.
- per_sequence_metrics.csv: per-sequence metrics computed from archived predictions.
- prediction_provenance.json: sequence names, GT and prediction SHA-256 hashes, and source collection identifiers. No personal absolute paths are included.
- cross_dataset_threshold_curves.csv: archived common-threshold curves used by the manuscript figures.
- cross_dataset_per_sequence_auc.csv: archived per-sequence figure data.
- aggregate_comparison.csv: original four-variant controlled comparison on UAV2UAV, including the unsuccessful direct-write dual-memory variant.
- a2a_multitracker_aggregate_metrics.csv: original A2A comparison export. A 2026 tag on the proposed/internal methods is an experiment-year placeholder, not a publication claim; the new benchmark_metrics.csv uses "-" for these entries.

The README uses UAV2UAV as a display label; the corresponding source release is UAV2UAV-2 (49 sequences). A2A contains 23 sequences. Image/GT length alignment is explicit, first-frame initialization is included, and invalid GT boxes are excluded from metrics without resetting the tracker. Data and predictions are not distributed here.

## Timing

historical_fps is preserved for traceability only. In particular, the original SafeCL/RLSM instrumentation summed visual-probe and camera-motion time, rather than the complete wrapper call, and excluded initialization. It must not be directly equated to the new runner's CUDA-synchronized fps_wall_excluding_init_io. FPS also depends on hardware, software, caching and sequence mix.

## Statistical interpretation

The UAV2UAV collection was used during method development; its small improvements are exploratory rather than held-out generalization evidence. A2A used the frozen configuration and was statistically neutral relative to SUTrack. None of the aggregate CSVs alone establishes a calibrated uncertainty model or a formal safety guarantee.
