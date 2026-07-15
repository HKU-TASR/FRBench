# Changelog

All notable changes to FRBench are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Pretrained weights are versioned separately via `weights-v*` release tags (see [`RELEASING.md`](./RELEASING.md)).

## [1.1.0] - 2026-07-15

### Added

- `frbench.FaceDetector`: a standalone, public face detector independent of any `FR` pipeline. Same input contract as `FR` (RGB float `[0, 255]`), lazy weight download, and `detect()` / `align()` / `crop()` methods returning un-normalized `[0, 255]` crops.
- Detector sharing: `FR(..., detector=face_detector_instance)` reuses one set of detector weights across multiple pipelines.
- Public differentiable geometry utilities: `frbench.align`, `frbench.crop`, `frbench.estimate_similarity_transform`, `frbench.invert_similarity`, `frbench.ARCFACE_112_TEMPLATE`, and `frbench.arcface_template(size)` for scaling the ArcFace template to other crop sizes.
- `FRDetectResult` accessors `boxes` / `scores` / `landmarks`, `num_images`, truthiness, and a documented 16-column layout; new builder `FRDetectResult.from_landmarks(...)` for feeding third-party detector outputs into `FR(..., detections=...)`.
- Unified configuration system: `frbench.configure(...)` (with `persist=True` writing `~/.frbench/config.json`) and the `frbench.configure_scoped(...)` context manager for temporary, thread-safe overrides. Precedence: defaults < config file < environment variables < runtime < scoped overrides.
- New environment variables `FRBENCH_WARNINGS`, `FRBENCH_DOWNLOAD_VERBOSE`, `FRBENCH_UPDATE_CHECK`, and `FRBENCH_CONFIG` (config-file location).
- `persist=` keyword on `set_verbose`, `set_warnings`, `set_download_verbose`, and `set_update_check`.
- CLI configuration management: `frbench-download --set KEY=VALUE`, `--unset KEY`, and `--show-config`.
- Offline unit-test suite (`tests/`) covering configuration, result types, geometry math, and input validation; wired into the publish workflow.
- The publish workflow now also creates a GitHub Release for each `v*` tag with the built sdist and wheel attached as assets.

### Fixed

- Assigning to `frbench.CACHE` / `REPO` / `RELEASE` silently did nothing; they are now live read-only views of the resolved configuration (use `frbench.configure()` to change them).
- `FR.detect()` no longer downloads and builds the recognition backbone; it only loads the detector weights.
- `set_warnings()` no longer mutates the global `warnings` filter on every call (a single category-scoped filter is registered once at import).

### Changed

- `FR(..., detector=...)` accepts either an asset-name string (as before) or a `frbench.FaceDetector` instance. All existing call signatures keep working.

## [1.0.0] - 2026-07-12

### Added

- Initial public release: unified, end-to-end differentiable face-recognition pipeline (`frbench.FR`) with 45+ pretrained weights covering 25 backbones, 9 loss functions, and 3 training datasets.
- On-demand weight downloads from GitHub Releases with SHA-256 verification, `frbench-download` CLI, verbosity switches, and update checks.

[1.1.0]: https://github.com/HKU-TASR/FRBench/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/HKU-TASR/FRBench/releases/tag/v1.0.0
