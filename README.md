# Optical Drone Navigation

A GPS-denied drone localization system written in C++. It estimates a drone's position purely from
aerial video, by matching camera frames against satellite reference imagery — no GPS or IMU input
required, only the video stream itself.

## Overview

Drones and other autonomous platforms typically rely on GPS for positioning, which can be jammed,
spoofed, or simply unavailable. This project explores an alternative: recovering position by
matching what the camera sees against a pre-loaded satellite map, using classical computer vision
feature matching combined with probabilistic state estimation.

The system runs in two modes: synthetic simulations (flights over Jerusalem and Manhattan satellite
imagery, for controlled algorithm benchmarking) and real-world pipelines (actual drone video, and a
telemetry-labeled aerial imagery dataset with dense GPS ground truth for quantitative validation).

## Features

- **Multiple feature-matching algorithms** behind a common interface — ORB, SIFT, AKAZE, Lucas-Kanade
  optical flow, and a hybrid ensemble — selectable per run for direct comparison.
- **Modular retrieval/matching pipeline**: candidate-crop retrieval and geometric match verification
  are split behind independent interfaces (`IRetrievalStage`/`IMatchingStage`), so a retrieval or
  matching implementation can be swapped and benchmarked in isolation — a classical baseline is in
  place, with a learned retrieval backbone planned next.
- **Two position-fusion strategies**: a 4-state Kalman filter with adaptive, confidence-weighted
  noise and outlier rejection, and a particle filter maintaining multiple position hypotheses
  simultaneously rather than committing to a single belief.
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
Video frame  ->  Preprocessing (CLAHE, mask, GSD normalization)
             ->  Feature matching against a reference-crop grid (ORB / SIFT / AKAZE / Hybrid)
             ->  Position fusion (Kalman filter or particle filter)
             ->  Trajectory output (video overlay + CSV telemetry)
             ->  Ground-truth evaluation (position error, outlier rate, confidence)
```

Key modules: `ORBFeatureEstimator` / `SIFTFeatureEstimator` / `AKAZEFeatureEstimator` /
`HybridEstimator` (feature matching), `SplitPipelineEstimator` with `IRetrievalStage` /
`IMatchingStage` (the modular retrieval/matching split), `KalmanFilter` / `ParticleFilter` (position
fusion), `GlobalLocator` (initial-position search), `VideoProcessing` (the main per-frame pipeline),
`TelemetryImporter` / `GroundTruthAnnotator` (ground truth ingestion and evaluation).

## Tech Stack

C++20, OpenCV 4.x, CMake, POSIX threads. Python 3 for dataset tooling, GeoTIFF conversion, and the
ground-truth evaluation script.

## Validation and Results

The pipeline was tested against two independent real-world sources: broadcast drone footage with
manually-annotated ground truth, and a telemetry-labeled aerial imagery dataset with dense,
GPS-verified ground truth across multiple flights and content types. This quantitative validation
surfaced and fixed several real geo-referencing bugs — coordinate-system errors, projection
mismatches — that visual inspection alone would not have caught.

Current real-world accuracy is best described as coarse localization (several hundred meters to
low kilometers) rather than precision positioning. Systematic testing across descriptor choice,
reference-map resolution, position-fusion strategy, and per-stage retrieval/matching instrumentation
narrowed this to a specific, measured bottleneck: most error originates in candidate retrieval
(finding the right map region to compare against in the first place), not in verifying a match once
a candidate is found, or in position fusion. This points to a cross-domain image matching problem —
real aerial camera footage matched against satellite imagery captured at a different time and by a
different sensor — that remains an active research area, not an implementation defect. On synthetic
data, where drone and reference imagery share an identical source, the same pipeline performs well,
which helped isolate the real-world gap to content and domain differences rather than the matching
algorithms themselves.

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

Actively investigated rather than a finished product. The core pipeline, both fusion strategies, and
the evaluation harness are complete and working. A systematic bottleneck diagnosis — instrumenting
retrieval accuracy independently of match verification and fusion — pinpointed candidate retrieval,
not feature matching itself, as the dominant source of error. The pipeline is now being restructured
around that finding: retrieval and matching are split behind independent, swappable interfaces (see
Features), with a classical baseline validated end-to-end and a learned retrieval backbone the next
step.
