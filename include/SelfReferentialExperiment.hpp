#pragma once

#include "DatasetSamples.hpp"
#include "PositionEstimation.hpp"

// ---------------------------------------------------------------------------
// Same-domain self-referential diagnostic (see CLAUDE.md Investigation Log).
//
// Everything else tested this session only changed how existing ORB output
// gets FUSED over time (heading scoring, search-window prior, particle
// filter) -- never what it's matched AGAINST. This experiment isolates that
// other variable directly: instead of the satellite reference map, build the
// reference database from a subset of the flight's OWN drone-camera frames
// (every map_frame_stride-th frame, by ground-truth position), then query
// with the remaining, disjoint frames -- a genuine held-out test, not the
// same-video-for-both-phases test the external OPTICAL_NAVIGATION_EX1
// project appears to use (see Investigation Log for that review).
//
// Deliberately a standalone diagnostic, not a new pipeline mode: reuses
// ORBFeatureEstimator, EdgeProcessor::preprocessFrame(), and
// CoordinateUtils::calculateDistance() completely unmodified, no Kalman/
// particle-filter fusion (the question here is single-frame match quality
// against a same-domain reference, not fusion -- that's already tested
// separately). Writes CSV Files/video_telemetry_orb_selfref_<sample>.csv in
// the existing telemetry schema so the unmodified
// scripts/evaluate_ground_truth.py works with zero changes.
//
// algorithm must be ORB for now (the only estimator this has been validated
// against; see CLAUDE.md for why single-variable measurement matters here).
// ---------------------------------------------------------------------------
void runSelfReferentialExperiment(
    const DatasetSampleConfig& cfg,
    PositionAlgorithm algorithm,
    int map_frame_stride = 5
);
