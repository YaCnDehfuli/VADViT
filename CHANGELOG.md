# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-05

First public GitHub release of the VADViT reference implementation (JISA 2025, DOI 10.1016/j.jisa.2025.104200).

### Added

- Volatility-based VAD region extractor (`vadinfo` / `vaddump`), region categorization, snapshot selection, and consolidation.
- Process-level grid construction (feature / entropy / Markov channels) and dataset loader with an 80/10/10 split.
- ViT wrapper (`timm`) with binary and multi-class (family) heads, partial freeze, dynamic unfreeze, and SWA.
- Training entrypoint with a data-leakage sanity check; evaluation with ROC, confusion-matrix, and attention-overlay paths.
- Single-sample inference and region-to-patch inspection.
- House-style README, architecture SVG, published attention figure, citation metadata (`CITATION.cff`), and MIT license.
- Pinned `requirements.txt` (including volatility3, capstone, transformers).

### Changed

- Dataset loader replaced with the revision used for the paper submission (prior tree had a stale local loader).
- README rewritten to put published 99.2% / 92% results and the pipeline on the first screen (PR #1).

### Fixed

- Documentation of dataset access, citation, and license for the paper release.

[1.0.0]: https://github.com/YaCnDehfuli/VADViT/releases/tag/v1.0.0
