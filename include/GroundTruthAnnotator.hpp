#pragma once

#include <string>
#include "HaifaSamples.hpp"
#include "GlobalLocator.hpp"

// ---------------------------------------------------------------------------
// Interactive ground-truth annotation tool.
//
// For a given Haifa sample, opens the video (with a fixed crosshair at frame
// centre -- the drone's nadir ground position, since the footage is
// straight-down) alongside its satellite map. The user scrubs to a frame
// where a recognizable feature sits under the crosshair and clicks that same
// point on the map. Converted to lat/lng using the exact same
// CoordinateUtils / mpp / center parameters as the navigation pipeline, so
// the recorded ground truth is directly comparable to telemetry CSVs
// produced by processVideoNavigation().
//
// Output: "CSV Files/ground_truth_<sample_name>.csv"
//   Frame,Time_Sec,Lat,Lng,Label
// ---------------------------------------------------------------------------
void runGroundTruthAnnotator(const HaifaSampleConfig& cfg);

// ---------------------------------------------------------------------------
// Reads "CSV Files/ground_truth_<sample_name>.csv" (if it exists) and
// returns its earliest-frame point as an initialization estimate.
//
// Used to bypass GlobalLocator + the manual click-to-confirm UI during
// testing: once a verified starting position already exists from ground-
// truth annotation, re-deriving it automatically (GlobalLocator, often
// inaccurate) or manually (imprecise, different every run) only adds noise
// to the very thing being evaluated. Returns {success=false} if no ground
// truth file exists for this sample, so callers should fall back to the
// normal GlobalLocator + manual-selection flow in that case.
// ---------------------------------------------------------------------------
InitializationData readGroundTruthStart(const std::string& sample_name);
