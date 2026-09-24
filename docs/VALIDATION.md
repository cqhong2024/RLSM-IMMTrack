# Release validation

Validation date: 2026-09-24. This record distinguishes release checks from the older full benchmark experiments.

## Completed source and numerical checks

- 19 automated tests passed. Coverage includes IMM covariance/probability stability, low-confidence correction gating, box transforms, delayed memory admission, quarantine, causal shadow prediction, speculative-output isolation, both data layouts, annotation conversion, configuration checks, output protection and metric/CLI scoring.
- Python AST comparisons matched all 17 top-level algorithm/metric definitions against the experimental source: 6 in imm.py, 6 in safecl.py, 2 in tracker.py, and 3 in metrics.py. No tracking or metric formula was altered during extraction.
- The vendored SUTrack files were copied from the pinned commit without edits.
- A checkpoint SHA-256 was computed from the actual experiment weight file.
- The packaged scorer re-evaluated archived prediction files for six methods on BOTH datasets: 12 complete method/dataset combinations, 432 sequence-result files. The primary AUC/P20 values agree with the manuscript exports. No network inference was rerun over all 72 sequences as part of packaging.
- The plot command was exercised and its rendered output inspected. Comparison plots enforce matching dataset/sequence/frame inventories.

## Actual CUDA inference smoke tests

Python 3.9.25, PyTorch 1.11.0+cu113, NVIDIA GeForce GTX 1080.

| Test | Sequence | Frames | Maximum absolute box-coordinate difference from archived predictions |
|---|---|---:|---:|
| RLSM / UAV2UAV | test4_DJI_0101_3 | 50 | 0.000242 px |
| SUTrack / UAV2UAV | test4_DJI_0101_3 | 8 | 0.000016 px |
| SafeCL / UAV2UAV | test4_DJI_0101_3 | 8 | 0.000035 px |
| RLSM / A2A | video01 | 8 | 0.000023 px |

The 50-frame RLSM test included actual shadow-selected outputs, not only an inactive baseline path. Coordinate comparisons used the saved xywh values, not just aggregate AUC. Tiny numerical differences are consistent with floating-point execution differences; these tests do not guarantee bitwise-identical trajectories on all devices.

A completed A2A smoke run was resumed with its original manifest; the sequence was hash-verified and skipped. The unit tests also verify that changed manifests and nonempty outputs cannot be silently reused.

## Archive checks

The archive-building audit checks for reparse points, private local paths, forbidden model/cache/data files, Python syntax errors, missing local Markdown links, missing license notices, and source/checksum mismatches. It extracts the ZIP into a separate directory and runs the test suite and command-line help from the extracted copy. The machine-readable archive verification report is delivered alongside the ZIP.

## Validation boundaries

This is a reproduction-oriented release candidate, not a fresh, independently held-out benchmark study. Full clean-room installation, Linux execution, new-GPU compatibility and statistically calibrated uncertainty were not established by these checks. Model weights and datasets must be obtained separately. Original-code licensing remains an author decision.
