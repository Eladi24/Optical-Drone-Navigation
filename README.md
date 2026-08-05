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
`HybridEstimator` (feature matching), `KalmanFilter` / `ParticleFilter` (position fusion),
`GlobalLocator` (initial-position search), `VideoProcessing` (the main per-frame pipeline),
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
reference-map resolution, position-fusion strategy, and search strategy narrowed this to a
structural limitation of matching real aerial camera footage against satellite imagery captured at
a different time and by a different sensor — a cross-domain image matching problem that remains an
active research area, not an implementation defect. On synthetic data, where drone and reference
imagery share an identical source, the same pipeline performs well, which helped isolate the
real-world gap to content and domain differences rather than the matching algorithms themselves.

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
the evaluation harness are complete and working; closing the real-world accuracy gap is ongoing,
with the most promising next steps being a same-domain reference map (built from drone imagery
rather than satellite tiles) and learned, domain-adapted feature matching in place of classical
descriptors.
