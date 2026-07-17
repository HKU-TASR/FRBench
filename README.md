# FRBench

An easy-to-use, **end-to-end differentiable** face-recognition module that ships **45+** pretrained weights,

- covering **25 backbones**, **9 loss functions**, and **3 training datasets** of different **types**,
- spanning **generations since 2015**

It is designed for **anti-facial-recognition (AFR)** research, where the transferability of attacks across backbones, loss functions, and training datasets is a central concern. But we believe that it benefits the face recognition field just as much.

Hand it any RGB image tensor (aligned or not); it detects and aligns the face(s), runs the chosen pretrained backbone, and returns embeddings ready for cosine similarity. Because the whole pipeline (detect &rarr; crop &rarr; align &rarr; backbone) is written in pure, differentiable PyTorch, **gradients flow from the embeddings all the way back to the input pixels**, enabling adversarial attacks and other gradient-based methods.

## Quickstart

Install with `pip install frbench` (or `pip install "frbench[demo]"` for the notebook extras).

```python
import frbench
from frbench import FR
import torch

device = torch.device("cpu") # cpu, cuda, mps
img = torch.rand(3, 480, 640, device=device) * 255.0 # RGB, [0, 255], float32 tensor
fr = FR("mobilevitv3-s", "arcface", "ms1m").to(device)  # (backbone, loss, dataset)
result = fr(img, l2_normalize=True)          # FREmbedResult(embeddings, indices, crops)
print(result.embeddings.shape)               # (1, 512) when one face is found

for m in frbench.list_models():
    print(m.backbone, m.loss, m.dataset) # available models
```

## Table of Contents

- [Highlights](#highlights)
- [Motivation](#motivation)
- [Installation](#installation)
- [Usage](#usage)
- [Model Zoo](#model-zoo)
- [Acknowledgement](#acknowledgement)
- [License](#license)
- [Citation](#citation)

## Highlights

- **Unified & consistent.** One module, one API, one weight format for 45+ models &mdash; no more stitching together incompatible repos.
- **Differentiable end-to-end.** Detection is non-differentiable, but cropping and 5-point alignment use `grid_sample` and a closed-form similarity transform, so the embedding is differentiable w.r.t. the raw input image.

## Motivation

In anti-facial-recognition (AFR) research, one needs to understand how well an attack transfers across face-recognition backbones, loss functions, and training datasets of different **types** and **generations**. For instance, training an attack on `ResNet50` and evaluating it on `SwinV2-T` (same loss, same data) fairly measures generalization to a next-generation architecture; training on `IRSE50` and evaluating on `IRSE100` isolates the effect of model size.

However, existing model zoos such as [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe) and [FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo) do not offer this controlled variation over backbones, losses, and datasets. Researchers end up collecting and deploying models from many libraries, and still cannot be confident that the "Swin Transformer" in repo A is the same as the "SwinV1" in repo B.

FRBench addresses this with a **single, unified, differentiable, easy-to-use PyTorch module** backed by a rich set of consistently-trained pretrained weights. (The training code may be released later.)

## Installation

### Dependencies

```bash
conda create -n frbench python=3.12
conda activate frbench
pip install frbench                 # core: torch, torchvision, pyyaml, tqdm
pip install "frbench[demo]"         # optional: pillow, matplotlib, seaborn
```

The core module needs `torch`, `torchvision`, `pyyaml`, and `tqdm` (for download progress bars). The demo notebook additionally uses `pillow`, `matplotlib`, and `seaborn`.

For local development, clone the repository and run `pip install -e ".[demo]"`.

### Weights download

Weights are downloaded on demand into `~/.frbench` (see [Notes on Configuration](#notes-on-configuration) to change the cache location). Prefetch assets with the CLI:

```bash
frbench-download --list                         # show all manifest keys
frbench-download --list-models                  # FR models only (no detectors)
frbench-download mobilefacenet_arcface_ms1m     # download one model
frbench-download --all                          # download everything
frbench-download --all --quiet                  # no progress / logs
frbench-download --refresh --list               # re-fetch manifest, then list
```

The public weights are downloaded directly from GitHub Releases; no GitHub login or token is required.

Progress bars use ``tqdm`` and are controlled by `set_download_verbose(bool)` (or `--quiet` on the download script); see [Notes on verbosity](#notes-on-verbosity).

## Usage

See [the demo notebook](./demo.ipynb) for a complete, runnable walkthrough.

### Face Embedding

1. **Input contract:** (list of) *RGB* and *[0, 255]* float tensors. The preprocessor warns if values look normalized. Can be of shape:

    - `(3, H, W)` — single image
    - `(N, 3, H, W)` — batch of N images
    - A list of `(3, Hi, Wi)` or `(1, 3, Hi, Wi)` tensors — variable-size images

2. **Return type:** `forward` / `embed` return a `frbench.FREmbedResult` named tuple, with fields accessible by name, by unpacking, and via `_asdict()`.

    | Field | Shape | Meaning |
    | --- | --- | --- |
    | `embeddings` | `(M, D)` | Face embeddings; empty `(0, D)` when no faces (never `None`). |
    | `indices` | `len M` | `indices[i]` is the source-image index of `embeddings[i]`. |
    | `crops` | `(M, 3, H, W)` | Normalized crops fed to the backbone. |

    `result.num_faces` gives `M`, and the result is truthy iff at least one face was produced.

3. **Key options:** (see the docstring for the full list):

    - `need_crop=False` — inputs are already-aligned 112×112 crops; skip detection.
    - `need_align=False` — crop a (loosened, see `loosen_crop`) square box instead of 5-point aligning.
    - `keep_largest=False` — return an embedding for every detected face, not just the largest.
    - `discard_invalid=True` — drop images with no detected face (default: fall back to the whole image).
    - `tta=("flip_horizontal",)` — test-time augmentations to average over (**default: horizontal flip; doubles backbone cost**). Pass `tta=()` to disable.
    - `l2_normalize=True` — L2-normalize the returned embeddings for direct cosine comparison.
    - `eager_load=True` (constructor) — download weights and build the backbone at init instead of first use.
    - `detections=` — pass precomputed detections from `FR.detect()` to skip re-detection

### Face Detection

Two methods are available for detection, cropping, and alignment: `FR.detect()` and `frbench.FaceDetector`. The former is a convenience wrapper that builds a detector on the fly; the latter is a standalone detector that can be shared across multiple `FR` pipelines so its weights load only once — useful when benchmarking attacks across many backbones:

```python
detector = frbench.FaceDetector().to(device)
frs = [frbench.FR(b, "arcface", "ms1m", detector=detector).to(device)
       for b in ("irse-100", "swinv2-b", "mobilefacenet")]

dets = detector.detect(imgs)                   # detect once
results = [fr(imgs, detections=dets) for fr in frs]
```

Note that `FR.detect()` doesn't build the recognition backbone — it only loads the detector weights.

1. **Input contract:** same as [Face Embedding](#face-embedding) above.
2. **Return type:** `FR.detect` / `FaceDetector.detect` return a `frbench.FRDetectResult` whose `detections` field holds one entry per image — `None` (no face) or a `(K, 16)` tensor with columns:

    | Columns | Meaning |
    | --- | --- |
    | 0-3 | Box `x1, y1, x2, y2` in input pixels. |
    | 4 | Background probability (`1 - face probability`). |
    | 5 | Face probability (detection confidence). |
    | 6-15 | Five landmarks `x1, y1, ..., x5, y5` in input pixels: left eye, right eye, nose, left mouth corner, right mouth corner. |

    You should rarely need to index those columns by hand: use the `dets.boxes`, `dets.scores`, and `dets.landmarks` accessors (per-image lists of `(K, 4)`, `(K,)`, and `(K, 5, 2)` tensors).

    To bring detections from **your own** face detector, build the named tuple with `FRDetectResult.from_landmarks` and pass it to `FR(..., detections=...)`:

    ```python
    import torch
    from frbench import FRDetectResult

    # Landmark order: left eye, right eye, nose, left/right mouth corners.
    ldm = torch.tensor([[38., 52.], [74., 52.], [56., 72.], [42., 92.], [71., 92.]])
    dets = FRDetectResult.from_landmarks([ldm, None])   # image 0: one face; image 1: none
    result = fr([img0, img1], detections=dets)          # boxes/scores default sensibly
    ```

### Face Cropping and Alignment

`FaceDetector.align` returns one `(K, 3, H, W)` tensor per source image.
Aligned faces can be reprojected into source coordinates with `unalign`. Request
the optional masks when compositing an edited face over its original image:

```python
dets = detector.detect(img)
aligned = detector.align(img, dets)

result = detector.unalign(
    aligned,
    dets,
    output_sizes=img.shape[-2:],
    return_mask=True,
)
canvases, masks = result

# Each face has its own source-sized canvas and boolean (1, H, W) mask.
merged_face = torch.where(masks[0][0], canvases[0][0], img)
```

Faces remain separate rather than being merged automatically, so applications
can choose their own overlap and blending policy. Unalignment reverses the
coordinates, not the information loss from cropping and interpolation; pixels
outside the mask come directly from the original image in the example above.

The differentiable geometry primitives are also available as standalone
functions:

```python
frbench.ARCFACE_112_TEMPLATE            # canonical (5, 2) ArcFace template, 112x112
frbench.arcface_template((224, 224))    # template scaled to another crop size
frbench.align(img, landmarks, template) # differentiable 5-point alignment
result = frbench.unalign(
    aligned_faces,
    landmarks,
    (source_h, source_w),
    template,
    return_mask=True,
)
frbench.crop(img, boxes)                # differentiable box crop + anti-aliased resize
frbench.estimate_similarity_transform(landmarks, template)  # (F, 2, 3) closed-form fit
frbench.invert_similarity(matrix)
```

### Notes on devices

Inference runs natively on CUDA, MPS, and CPU. For **backprop on Apple MPS**, PyTorch needs a CPU fallback for a couple of ops (e.g. `grid_sample`'s backward); the package sets `PYTORCH_ENABLE_MPS_FALLBACK=1` on import, which is honored as long as the package is imported **before** `torch`. If you import `torch` first, set it yourself beforehand:

```python
import os; os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
```

### Notes on configuration

All settings go through one resolution chain, from lowest to highest precedence:

1. Built-in defaults
2. The persisted config file `~/.frbench/config.json` (location overridable with `FRBENCH_CONFIG`)
3. Environment variables
4. Runtime calls to `frbench.configure(...)` (or `set_verbose` and friends)
5. Active `frbench.configure_scoped(...)` blocks (innermost wins)

| Setting | Env var | Default |
| --- | --- | --- |
| `cache` — weights directory | `FRBENCH_CACHE` | `~/.frbench` |
| `repo` — GitHub repo | `FRBENCH_REPO` | `HKU-TASR/FRBench` |
| `release` — release tag | `FRBENCH_RELEASE` | `weights-v1.0.0` |
| `warnings` — inference warnings | `FRBENCH_WARNINGS` | `true` |
| `download_verbose` — download logs/bars | `FRBENCH_DOWNLOAD_VERBOSE` | `true` |
| `update_check` — update reminders | `FRBENCH_UPDATE_CHECK` | `true` |

```python
import frbench

# For this process only:
frbench.configure(cache="/data/frbench", verbose=False)

# Permanently (written to ~/.frbench/config.json, applies to future processes):
frbench.configure(cache="/data/frbench", verbose=False, persist=True)

# Temporarily, inside a with-block (thread-safe; restored on exit):
with frbench.configure_scoped(verbose=False, cache="/tmp/frbench"):
    fr = frbench.FR("mobilefacenet", "arcface", "ms1m")

# Reset a setting to its default (and remove it from the config file):
frbench.configure(cache=None, persist=True)
```

The same persistence is available from the command line:

```bash
frbench-download --set cache=/data/frbench --set download_verbose=false
frbench-download --unset cache
frbench-download --show-config      # resolved values and where each comes from
```

Module-level `frbench.CACHE`, `frbench.REPO`, and `frbench.RELEASE` are live read-only views of the resolved values (change them with `configure()` — assigning to them has no effect).

The face detector defaults to `retinaface_mobilenetv1` and can be changed per model via `FR(..., detector="retinaface_resnet50")`, or shared across models by passing a `frbench.FaceDetector` instance.

### Notes on verbosity

Output has two independent switches, both enabled by default and toggleable on the fly:

```python
import frbench

frbench.set_warnings(False)          # silence inference warnings (e.g. no face detected)
frbench.set_download_verbose(False)  # silence download status + progress bars
frbench.set_verbose(False)           # convenience: both at once

frbench.set_verbose(False, persist=True)   # persist across processes
with frbench.configure_scoped(verbose=False):
    ...                                    # silence temporarily
```

- **Inference warnings** (no face detected, unsupported TTA, etc.) go through Python's standard `warnings` module under the `frbench.FRBenchWarning` category, so you can also filter or capture them with the usual tools, e.g. `warnings.filterwarnings("ignore", category=frbench.FRBenchWarning)` or `with warnings.catch_warnings(record=True) as w: ...`.
- **Download verbosity** (log messages and the `tqdm` progress bar) is a separate channel controlled by `set_download_verbose` (or `--quiet` on `frbench-download`).
- **Critical problems** (invalid input tensors, missing manifest keys, failed downloads) raise exceptions (`ValueError`, `frbench.FRBenchDownloadError`, etc.) regardless of the switches above.

### Notes on update checks

On the first `FR(...)` construction or `frbench-download` command in a process, FRBench checks whether a newer package or weights release is available. Results are cached for 24 hours, network failures are ignored, and no check runs merely from importing the package.

Set `FRBENCH_UPDATE_CHECK=0` (the legacy `FRBENCH_NO_UPDATE_CHECK=1` also works) or call `frbench.set_update_check(False)` (add `persist=True` to make it permanent) to disable these reminders. Installed versions remain pinned to the weights release they were tested with; updating is always explicit.

## Model Zoo

Every model is addressed by the same three manifest keys you pass to `FR(backbone_type, loss_type, dataset_type)`. All accuracies below are in percent.

| Name | `backbone_type` | `loss_type` | `dataset_type` | LFW | CPLFW | CALFW | CFP-FP | AgeDB-30 | MegaFace (1M; Rank-1 Id) | MegaFace (1M; TAR@FAR=1e-4 Ver) | IJB-C (N1D1F1; TAR@FAR=1e-4 Ver) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet100-ArcFace-MS1M | `resnet-100` | `arcface` | `ms1m` | 99.75 | 92.18 | 95.73 | 97.3 | 96.37 | 93.19 | 98.49 | 95.21 |
| ResNetV2_100-ArcFace-MS1M | `resnetv2-100` | `arcface` | `ms1m` | 99.83 | 93.6 | 95.77 | 98.29 | 97.2 | 96.72 | 99.26 | 95.5 |
| DenseNet201-ArcFace-MS1M | `densenet-201` | `arcface` | `ms1m` | 99.85 | 93.42 | 96 | 98.56 | 97.92 | 97.75 | 99.38 | 95.91 |
| IRSE100-ArcFace-MS1M | `irse-100` | `arcface` | `ms1m` | 99.77 | 93.08 | 95.87 | 98.3 | 97.7 | 97.48 | 99.61 | 96.31 |
| EfficientNet_b4-ArcFace-MS1M | `efficientnetv1-b4` | `arcface` | `ms1m` | 99.73 | 91.83 | 95.72 | 97.33 | 96.95 | 95.89 | 99.08 | 94.98 |
| MobileFaceNet_ECA-ArcFace-MS1M | `mobilefacenet` | `arcface` | `ms1m` | 99.68 | 90.85 | 95.67 | 95.2 | 96.82 | 93.93 | 98.41 | 93.23 |
| SwinV1_B-ArcFace-MS1M | `swinv1-b` | `arcface` | `ms1m` | 99.8 | 93.7 | 96.05 | 98.64 | 98.03 | 98.8 | 99.56 | 96.8 |
| ConvNeXt_B-ArcFace-MS1M | `convnext-b` | `arcface` | `ms1m` | 99.83 | 93.7 | 96.02 | 98.79 | 97.75 | 98.61 | 99.49 | 96.35 |
| ConvNeXtV2_B-ArcFace-MS1M | `convnextv2-b` | `arcface` | `ms1m` | 99.8 | 93.82 | 96.03 | 98.8 | 97.65 | 98.63 | 99.51 | 96.41 |
| SwinMLP_B-ArcFace-MS1M | `swinmlp-b` | `arcface` | `ms1m` | 99.82 | 93.27 | 96.22 | 98.69 | 97.95 | 98.86 | 99.55 | 96.78 |
| SwinV2_B-ArcFace-MS1M | `swinv2-b` | `arcface` | `ms1m` | 99.82 | 93.48 | 96.25 | 98.69 | 97.88 | 98.67 | 99.54 | 96.49 |
| MobileViTV1_S-ArcFace-MS1M | `mobilevit-s` | `arcface` | `ms1m` | 99.5 | 90.47 | 95.43 | 94.71 | 95.67 | 90.83 | 97.74 | 92.82 |
| MobileViTV2_2.0-ArcFace-MS1M | `mobilevitv2-2.0` | `arcface` | `ms1m` | 99.5 | 91.03 | 95.58 | 96.24 | 96.23 | 94.62 | 98.69 | 93.97 |
| MobileViTV3V1_S-ArcFace-MS1M | `mobilevitv3-s` | `arcface` | `ms1m` | 99.68 | 90.95 | 95.53 | 95.21 | 96.27 | 92.22 | 98.08 | 93.4 |
| MobileViTV3V2_2.0-ArcFace-MS1M | `mobilevitv3-2.0` | `arcface` | `ms1m` | 99.75 | 91.8 | 95.7 | 96.87 | 96.67 | 95.79 | 98.97 | 94.48 |
| MobileNetV1_W1-ArcFace-MS1M | `mobilenet-w1` | `arcface` | `ms1m` | 99.58 | 90.7 | 95.8 | 95.96 | 96.97 | 95.02 | 98.64 | 93.6 |
| MobileNetV2_W1-ArcFace-MS1M | `mobilenetv2-w1` | `arcface` | `ms1m` | 99.52 | 89.72 | 95.5 | 93.73 | 96.77 | 92.48 | 97.96 | 92.82 |
| MobileNetV3_L-ArcFace-MS1M | `mobilenetv3-l` | `arcface` | `ms1m` | 99.57 | 90.15 | 95.7 | 94.71 | 96.98 | 93.65 | 98.3 | 92.88 |
| MobileNetV4_Conv_M-ArcFace-MS1M | `mobilenetv4conv-m` | `arcface` | `ms1m` | 99.72 | 91.63 | 96.03 | 96.74 | 97.38 | 96 | 99.02 | 94.63 |
| SwinV2_T-ArcFace-MS1M | `swinv2-t` | `arcface` | `ms1m` | 99.78 | 92.67 | 96.05 | 97.99 | 97.58 | 97.92 | 99.42 | 96 |
| SwinV2_S-ArcFace-MS1M | `swinv2-s` | `arcface` | `ms1m` | 99.8 | 93.37 | 95.88 | 98.36 | 97.93 | 98.23 | 99.43 | 96.33 |
| SwinV2_L-ArcFace-MS1M | `swinv2-l` | `arcface` | `ms1m` | 99.78 | 93.95 | 96.05 | 98.81 | 98.18 | 98.84 | 99.58 | 96.78 |
| IRSE18-ArcFace-MS1M | `irse-18` | `arcface` | `ms1m` | 99.47 | 91.57 | 95.88 | 96.66 | 96.87 | 94.47 | 98.57 | 94.04 |
| IRSE34-ArcFace-MS1M | `irse-34` | `arcface` | `ms1m` | 99.8 | 92.78 | 95.95 | 97.93 | 97.72 | 96.92 | 99.24 | 95.63 |
| IRSE50-ArcFace-MS1M | `irse-50` | `arcface` | `ms1m` | 99.8 | 93.27 | 95.97 | 98.21 | 97.73 | 97.26 | 99.29 | 96.07 |
| IRSE100-Triplet-MS1M | `irse-100` | `tripletloss` | `ms1m` | 99.78 | 91.97 | 95.47 | 97.93 | 97.1 | 91.05 | 98.27 | 91.17 |
| IRSE100-Center-MS1M | `irse-100` | `centerloss` | `ms1m` | 99.77 | 92.98 | 95.92 | 98.49 | 97.32 | 96.24 | 99.32 | 93.59 |
| IRSE100-SphereFace-MS1M | `irse-100` | `sphereface` | `ms1m` | 99.8 | 93.95 | 96.23 | 98.97 | 98.33 | 98.18 | 99.49 | 96.6 |
| IRSE100-CosFace-MS1M | `irse-100` | `cosface` | `ms1m` | 99.8 | 93.9 | 96.07 | 98.89 | 98.28 | 98.83 | 99.59 | 97 |
| IRSE100-CurricularFace-MS1M | `irse-100` | `curricularface` | `ms1m` | 99.82 | 93.82 | 95.98 | 98.8 | 98.15 | 98.93 | 99.59 | 96.98 |
| IRSE100-MagFace-MS1M | `irse-100` | `magface` | `ms1m` | 99.82 | 93.67 | 95.98 | 98.86 | 98.47 | 98.94 | 99.58 | 96.89 |
| IRSE100-AdaFace-MS1M | `irse-100` | `adaface` | `ms1m` | 99.82 | 93.72 | 96.17 | 98.76 | 98.23 | 98.88 | 99.61 | 96.96 |
| IRSE100-UniFace-MS1M | `irse-100` | `uniface` | `ms1m` | 99.8 | 93.43 | 96.13 | 98.64 | 98.3 | 98.77 | 99.57 | 96.87 |
| SwinV2_B-Triplet-MS1M | `swinv2-b` | `tripletloss` | `ms1m` | 99.77 | 93.2 | 95.88 | 98.36 | 97.05 | 94.65 | 98.97 | 93.66 |
| SwinV2_B-Center-MS1M | `swinv2-b` | `centerloss` | `ms1m` | 99.55 | 89.93 | 93.92 | 97.4 | 95.57 | 93.75 | 97.54 | 82.23 |
| SwinV2_B-SphereFace-MS1M | `swinv2-b` | `sphereface` | `ms1m` | 99.85 | 93.8 | 96.28 | 98.47 | 97.82 | 97.71 | 99.38 | 96.28 |
| SwinV2_B-CosFace-MS1M | `swinv2-b` | `cosface` | `ms1m` | 99.82 | 93.8 | 96.25 | 98.63 | 97.8 | 98.36 | 99.49 | 96.63 |
| SwinV2_B-CurricularFace-MS1M | `swinv2-b` | `curricularface` | `ms1m` | 99.8 | 93.63 | 96.07 | 98.43 | 97.93 | 98.56 | 99.47 | 96.6 |
| SwinV2_B-MagFace-MS1M | `swinv2-b` | `magface` | `ms1m` | 99.8 | 93.87 | 96.07 | 98.54 | 97.95 | 98.62 | 99.5 | 96.5 |
| SwinV2_B-AdaFace-MS1M | `swinv2-b` | `adaface` | `ms1m` | 99.82 | 93.62 | 96 | 98.27 | 97.72 | 98.66 | 99.5 | 96.64 |
| SwinV2_B-UniFace-MS1M | `swinv2-b` | `uniface` | `ms1m` | 99.75 | 93.55 | 95.92 | 98.51 | 97.6 | 98.3 | 99.42 | 96.51 |
| IRSE100-ArcFace-WebFace4M | `irse-100` | `arcface` | `webface4m` | 99.68 | 92.97 | 95.4 | 98.3 | 96.12 | 95.19 | 98.88 | 95.13 |
| IRSE100-ArcFace-Glint360k | `irse-100` | `arcface` | `glint360k` | 99.8 | 94.07 | 95.88 | 98.81 | 97.62 | 98.25 | 99.47 | 96.71 |
| SwinV2_B-ArcFace-WebFace4M | `swinv2-b` | `arcface` | `webface4m` | 99.82 | 94.42 | 95.75 | 98.69 | 97.58 | 97.14 | 99.39 | 96.82 |
| SwinV2_B-ArcFace-Glint360k | `swinv2-b` | `arcface` | `glint360k` | 99.85 | 95.03 | 96.07 | 99.03 | 98.3 | 98.81 | 99.63 | 97.43 |

### Coverage of backbones

Under the same loss function (ArcFace) and the same dataset (MS1M), we have:

1. SOTA backbones from 2015-2024

    | Backbone | Year | Venue |
    | --- | --- | --- |
    | [ResNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | 2015 | NIPS |
    | [ResNetV2](https://arxiv.org/pdf/1603.05027) | 2016 | ECCV |
    | [DenseNet](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) | 2017 | CVPR |
    | [SE-Net](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)(IRSE) | 2018 | CVPR |
    | [EfficientNet](https://proceedings.mlr.press/v97/tan19a/tan19a.pdf) | 2019 | ICML |
    | [ECA-Net](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf)(MobileFaceNet_ECA) | 2020 | CVPR |
    | [SwinV1](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf) | 2021 | ICCV |
    | [ConvNeXtV1](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf) | 2022 | CVPR |
    | [ConvNeXtV2](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf) | 2023 | CVPR |
    | [MobileNetV4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05647.pdf) | 2024 | ECCV |

2. Backbones of the same family across generations

    | Backbone | Family | Paradigm |
    | --- | --- | --- |
    | SwinMLP | Swin Transformer | Transformer |
    | SwinV1 | Swin Transformer | Transformer |
    | SwinV2 | Swin Transformer | Transformer |
    | MobileViTV1 | MobileViT | Transformer |
    | MobileViTV2 | MobileViT | Transformer |
    | MobileViTV3V1 | MobileViT | Transformer |
    | MobileViTV3V2 | MobileViT | Transformer |
    | ConvNeXtV1 | ConvNeXt | CNN |
    | ConvNeXtV2 | ConvNeXt | CNN |
    | MobileNetV1 | MobileNet | CNN |
    | MobileNetV2 | MobileNet | CNN |
    | MobileNetV3 | MobileNet | CNN |
    | MobileNetV4-Conv | MobileNet | CNN |

3. Backbones of the same type across sizes

    | Backbone | Type | Parameter Size (M) | Paradigm |
    | --- | --- | --- | --- |
    | SwinV2-T | SwinV2 | 47.099 | Transformer |
    | SwinV2-S | SwinV2 | 68.57 | Transformer |
    | SwinV2-B | SwinV2 | 112.926 | Transformer |
    | SwinV2-L | SwinV2 | 234.078 | Transformer |
    | IRSE18 | IRSE | 24.115 | CNN |
    | IRSE34 | IRSE | 34.303 | CNN |
    | IRSE50 | IRSE | 43.824 | CNN |
    | IRSE100 | IRSE | 65.549 | CNN |

### Coverage of losses

Under the same backbone (IRSE100 for CNN, SwinV2-B for Transformer) and the same dataset (MS1M), we have SOTA loss functions from 2015-2023

| Loss | Year | Venue |
| --- | --- | --- |
| [Triplet Loss](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) | 2015 | CVPR |
| [Center Loss](https://kpzhang93.github.io/papers/eccv2016.pdf) | 2016 | ECCV |
| [SphereFace](https://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf) | 2017 | CVPR |
| [CosFace/LMCL](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_CosFace_Large_Margin_CVPR_2018_paper.pdf) | 2018 | CVPR |
| [ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf) | 2019 | CVPR |
| [CurricularFace](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_CurricularFace_Adaptive_Curriculum_Learning_Loss_for_Deep_Face_Recognition_CVPR_2020_paper.pdf) | 2020 | CVPR |
| [MagFace](https://openaccess.thecvf.com/content/CVPR2021/papers/Meng_MagFace_A_Universal_Representation_for_Face_Recognition_and_Quality_Assessment_CVPR_2021_paper.pdf) | 2021 | CVPR |
| [AdaFace](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.pdf) | 2022 | CVPR |
| [UniFace](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_UniFace_Unified_Cross-Entropy_Loss_for_Deep_Face_Recognition_ICCV_2023_paper.pdf) | 2023 | ICCV |

### Coverage of datasets

Under the same backbone (IRSE100 for CNN, SwinV2-B for Transformer) and the same loss function (ArcFace), we have 3 datasets of varying image and identity counts:

| Dataset | Images | Identities |
| --- | --- | --- |
| [MS1M](https://github.com/ZhaoJ9014/face.evoLVe) | 5,822,653 | 85,742 |
| [WebFace4M](https://github.com/HaiyuWu/vec2face) | 4,235,242 | 205,990 |
| [Glint360k](https://github.com/deepinsight/insightface) | 17,091,657 | 360,232 |

## Acknowledgement

Codes in [`frbench/backbones/`](./frbench/backbones/) and [frbench/utils/retinaface.py](./frbench/utils/retinaface.py) are adapted from existing open-source projects:

- [timm](https://github.com/huggingface/pytorch-image-models)
- [cavaface](https://github.com/cavalleria/cavaface)
- [FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo)
- [InsightFace](https://github.com/deepinsight/insightface)
- [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- [facebookresearch/ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)
- [apple/ml-cvnets](https://github.com/apple/ml-cvnets)
- [micronDLA/MobileViTv3](https://github.com/micronDLA/MobileViTv3)
- [jaiwei98/mobile-vit-pytorch](https://github.com/jaiwei98/mobile-vit-pytorch)
- [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [HKU-TASR/Protego](https://github.com/HKU-TASR/Protego)

## License

This project is licensed under the [MIT License](./LICENSE).

## Citation

If you find our project useful in your research, please consider citing:

```bibtex
@inproceedings{wang2026protego,
    title={Protego: User-Centric Pose-Invariant Privacy Protection Against Face Recognition-Induced Digital Footprint Exposure},
    author={Ziling Wang and Shuya Yang and Jialin Lu and Ka-Ho Chow},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2026}
}
```
