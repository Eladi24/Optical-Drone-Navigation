#pragma once

#include <string>

// ---------------------------------------------------------------------------
// Assembles an ordered sequence of UAV-VisLoc drone/<prefix>_%04d.JPG images
// into a single synthetic video file -- a mechanical convenience so
// VideoPreprocessor (mask detection, altitude estimation: neither cares about
// real inter-frame timing) and processVideoNavigation's cv::VideoCapture-based
// frame loop work completely unchanged for a dataset that ships discrete
// images rather than a video file.
//
// IMPORTANT: this video's own frame rate is a nominal constant with NO
// relation to the dataset's real, uneven capture intervals (UAV-VisLoc's
// survey-flight turnarounds take far longer between frames than straight
// runs). Never treat this video's fps as physically meaningful -- real
// per-frame timestamps come from TelemetryImporter's telemetry_<sample>.csv
// and must be fed to processVideoNavigation() via its frame_times_sec
// parameter instead.
//
// image_dir must contain files named "<prefix>_%04d.JPG" for frame_count
// consecutive images, 1-indexed (UAV-VisLoc's own convention). Cached to
// output_path the same way Haifa's preprocessed video is -- if output_path
// already exists this is a no-op; delete it to force reassembly.
// ---------------------------------------------------------------------------
bool assembleDatasetVideo(const std::string& image_dir,
                          const std::string& prefix,
                          int frame_count,
                          const std::string& output_path);
