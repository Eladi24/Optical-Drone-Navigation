# Optical Drone Navigation

A GPS-denied drone localization system written in C++. It estimates a drone's position from its
downward-looking camera, by matching camera frames against satellite reference imagery. No GPS is
used; the only other inputs are the standard non-GNSS sensors a drone already carries (altitude,
attitude, compass), and the core matching pipeline runs from imagery alone.

## Overview

Drones and other autonomous platforms typically rely on GPS for positioning, which can be jammed,
spoofed, or simply unavailable. This project explores an alternative: recovering position by
matching what the camera sees against a pre-loaded satellite map, combining classical and learned
computer vision (transformer-based place recognition, learned local-feature matching) with
probabilistic and trajectory-level state estimation.

The system runs in two modes: synthetic simulations (flights over Jerusalem and Manhattan satellite
imagery, for controlled algorithm benchmarking) and real-world pipelines (actual drone video, and a
telemetry-labeled aerial imagery dataset with dense GPS ground truth for quantitative validation).

## Features

- **Multiple feature-matching algorithms** behind a common interface — ORB, SIFT, AKAZE, Lucas-Kanade
  optical flow, and a hybrid ensemble — selectable per run for direct comparison.
- **Modular retrieval/matching pipeline**: candidate-crop retrieval and geometric match verification
  are split behind independent interfaces (`IRetrievalStage`/`IMatchingStage`), so a retrieval or
  matching implementation can be swapped and benchmarked in isolation. Classical (color histogram +
  ORB/RANSAC) and learned (frozen and fine-tuned transformer retrieval, XFeat and DISK+LightGlue
  local matching, via ONNX Runtime) implementations are both in place and directly comparable.
- **Three position-fusion strategies**: a 4-state Kalman filter with adaptive, confidence-weighted
  noise and outlier rejection; a particle filter maintaining multiple position hypotheses
  simultaneously rather than committing to a single belief; and an offline trajectory-level batch
  estimator that exploits whole-flight structure — correct matches cluster near the true path even
  when many individual frames are unreliable — to recover signal a purely causal filter cannot.
- **Relative-motion estimation**: camera-only odometry from consecutive frames (same-domain
  registration, scaled by altitude and oriented by compass/attitude), combined with the per-frame
  map matches through a local consensus step that keeps only matches agreeing with the flown
  trajectory; plus an integrated visual-inertial odometry module (camera + IMU), evaluated against a
  second, IMU-equipped ground-truth dataset.
- **Real-world video pipeline**: automatic on-screen-overlay masking, altitude estimation from
  reference objects, ground-sample-distance normalization so drone and reference imagery are matched
  at consistent real-world scale.
- **Dataset validation pipeline**: ingests a telemetry-labeled aerial imagery dataset (UAV-VisLoc),
  with dense per-frame GPS/altitude/heading ground truth, satellite GeoTIFF processing, and automatic
  ground-truth-based initialization.
- **Ground-truth evaluation harness**: quantitative position-error scoring against real or
  manually-annotated ground truth, rather than relying on visual inspection.
- **Multi-threaded C++ core**: a persistent POSIX thread pool parallelizes per-frame feature matching
  against the reference database.

## Architecture

```
Camera frame ->  Preprocessing (CLAHE, mask, ground-sample-distance normalization)
             ->  Retrieval: shortlist the top-k satellite crops
                 (DINOv2 + VLAD embeddings, or classical ORB / SIFT / AKAZE search)
             ->  Matching: verify each candidate geometrically, recover the position
                 (DISK + LightGlue, XFeat, or ORB + RANSAC)
             ->  Fusion: Kalman / particle filter (online), or trajectory-level
                 batch estimation with frame-to-frame odometry (offline)
             ->  Output: video overlay, CSV telemetry, ground-truth evaluation
```

Key modules: `ORBFeatureEstimator` / `SIFTFeatureEstimator` / `AKAZEFeatureEstimator` /
`HybridEstimator` (feature matching), `SplitPipelineEstimator` with `IRetrievalStage` /
`IMatchingStage` (the modular retrieval/matching split), `OnnxVladRetrieval` / `OnnxDeitRetrieval`
(learned retrieval), `DiskLightGlueMatching` / `XFeatMatching` (learned matching),
`KalmanFilter` / `ParticleFilter` (position
fusion), `GlobalLocator` (initial-position search), `VideoProcessing` (the main per-frame pipeline),
`TelemetryImporter` / `GroundTruthAnnotator` (ground truth ingestion and evaluation).

## Learned Models

All learned components run inside the C++ pipeline through ONNX Runtime; none require a GPU at
inference time.

- **Retrieval backbone: DINOv2 ViT-S/14** (self-supervised vision transformer, 21M parameters). A
  frozen DeiT-Tiny serves as the ImageNet-pretrained baseline. DINOv2 was fine-tuned for
  drone-to-satellite retrieval with a temperature-scaled InfoNCE contrastive loss: each drone frame
  is pulled toward the satellite crop at its true location and pushed away from mined look-alike
  crops (hard negatives), random distant crops, and other samples in the batch. Only the last two
  transformer blocks are trained, a deliberate choice for a dataset of a few thousand frames;
  training ran on a cloud GPU (Colab) on development and validation flights only, so held-out
  flights stay unseen.
- **VLAD aggregation**: instead of one pooled vector per image, the fine-tuned backbone's dense
  patch features (256 per image) are aggregated against a 64-word visual vocabulary learned by
  k-means (AnyLoc-style). This preserves spatial detail and gives the strongest retrieval on unseen
  flights.
- **Local matching: DISK + LightGlue** (learned keypoints with an attention-based matcher), with
  XFeat as a lighter alternative. Both are permissively licensed; SuperPoint/SuperGlue were
  deliberately avoided for licensing reasons given the embedded-deployment target.

## Python and C++

The two languages have separate roles. **C++ is the runtime system**: the per-frame pipeline,
model inference (ONNX Runtime), filtering, and everything intended to run on the target hardware.
**Python is the research and tooling layer**: model fine-tuning and ONNX export (PyTorch/timm),
dataset preparation (GeoTIFF conversion, terrain-elevation lookup, telemetry import helpers), and
offline analysis (benchmarking, trajectory-level fusion, odometry prototypes).

They meet through files, not bindings: Python exports models (`.onnx`) and the VLAD vocabulary (a
compact binary) that the C++ pipeline loads; the C++ pipeline writes per-frame telemetry CSVs that
the Python evaluation and fusion scripts consume; and `scripts/run_benchmark.py` drives the C++
executable flight by flight. New ideas are prototyped and measured in Python first, then ported to
C++ once they prove their value. The VLAD retrieval stage followed exactly this path, from a Python
prototype to `OnnxVladRetrieval`. The OpenCV calls in the Python scripts use the same compiled
OpenCV library as the C++ code, so the split is about iteration speed and tooling, not raw
performance.

## Tech Stack

C++20, OpenCV 4.x, ONNX Runtime, CMake, POSIX threads. Python 3 (PyTorch, timm, NumPy/SciPy,
OpenCV) for model training and export, dataset tooling, and evaluation/benchmarking. Deployment
target: NVIDIA Jetson Orin (ONNX Runtime / TensorRT).

## Validation and Results

The pipeline was tested against two independent real-world sources: broadcast drone footage with
manually-annotated ground truth, and a telemetry-labeled aerial imagery dataset with dense,
GPS-verified ground truth across multiple flights and content types. This quantitative validation
surfaced and fixed several real geo-referencing bugs — coordinate-system errors, projection
mismatches — that visual inspection alone would not have caught.

Current real-world accuracy is best described as coarse localization (several hundred meters to
low kilometers) rather than precision positioning. Systematic testing across descriptor choice,
reference-map resolution, position-fusion strategy, and per-stage retrieval/matching instrumentation
shows that a large share of the error starts at candidate retrieval: finding the right map region to
compare against in the first place. The underlying difficulty is cross-domain image matching. Real
aerial camera footage is matched against satellite imagery captured at a different time and by a
different sensor, which remains an active research area. On synthetic data, where drone and
reference imagery share an identical source, the same pipeline performs well, which helped isolate
the real-world gap to content and domain differences.

Per-stage profiling later exposed a silent bug in the learned matchers: a feature cache keyed by
image memory address meant every frame was verified against the first frame's keypoints, because
the frame buffer is reused at the same address. It is fixed, guarded by an invariant check, and the
learned-matching results are being re-measured flight by flight.

A trajectory-level fusion strategy — exploiting whole-flight structure rather than filtering frame by
frame — measurably reduces error beyond the causal Kalman filter alone, evidence that some of the
remaining gap is recoverable through better use of existing per-frame evidence, not only through
better per-frame matching. Adding camera-only odometry between consecutive frames, and keeping only
the map matches that agree with that odometry over a local window, reduces median error by roughly
35–80% relative to the earlier trajectory-level estimator on development flights.

Measuring the camera model directly from the data, rather than assuming it, improved the
drone-to-map scale alignment: the field of view is recovered from frame-to-frame registration, and
terrain-corrected altitude comes from a public elevation model (Copernicus DEM). A full
re-evaluation across all flights with the calibrated model is in progress; early results show
substantially higher retrieval accuracy on previously unseen flights.

The visual-inertial odometry module was separately evaluated against a second, IMU-equipped
ground-truth dataset. Its accuracy is currently limited on low-texture terrain, where sparse visual
features are insufficient to keep inertial drift in check — a further, concrete instance of the same
underlying theme: performance here is gated by scene content and feature discriminability, not by the
estimation algorithms themselves.

## Getting Started

```bash
cmake -S . -B build && cmake --build build -j
./build/Main orb h        # Haifa real-video pipeline, ORB
./build/Main sift j       # Jerusalem simulation, SIFT
./build/Main orb d        # Telemetry-labeled dataset validation, ORB
```

An API key is required for the live simulation/Haifa modes (Google Static Maps); copy
`config.ini.sample` to `config.ini`. The dataset validation mode needs no API key.

## Status

Actively investigated rather than a finished product. The core pipeline, all three fusion strategies,
and the evaluation harness are complete and working. A systematic bottleneck diagnosis, instrumenting
retrieval accuracy independently of match verification and fusion, identified candidate retrieval as
a major source of error. That motivated the retrieval/matching split and the learned (ONNX Runtime)
implementations now in place alongside the classical baseline. Trajectory-level batch fusion has
since been evaluated on held-out flights: it consistently reduces error but does not, on its own,
close the gap to precision positioning. Learned retrieval has since advanced from frozen to
fine-tuned to VLAD-aggregated transformer features. Current work: re-evaluating the learned pipeline
flight by flight after the camera-model and feature-cache fixes, combining frame-to-frame odometry
with map matches, and coarse-to-fine
registration (using the estimated trajectory as a prior for precise local alignment against the
map). Real-time deployment on Jetson Orin follows once accuracy targets are met.
