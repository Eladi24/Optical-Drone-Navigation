#pragma once
#include <opencv2/opencv.hpp>
#include <string>

// ---------------------------------------------------------------------------
// EdgeProcessor — frame preprocessing utilities for low-texture aerial video.
//
// Three responsibilities:
//   1. Build a HUD mask that zeros out burned-in telemetry overlays.
//   2. Normalize every frame via CLAHE before feature extraction or matching.
//   3. Produce Canny edge maps used for scale-aware template matching.
// ---------------------------------------------------------------------------
namespace EdgeProcessor {

// Compute a binary mask from the first n_frames of the video.
// Pixels with low temporal variance (< variance_threshold) are static
// HUD elements; the mask marks them as 0.  All other pixels are 255.
// Returns an empty Mat if the video cannot be opened.
cv::Mat computeTemporalHudMask(const std::string& video_path,
                                int   n_frames          = 30,
                                float variance_threshold = 8.0f);

// Simple fallback: erode a fixed margin from each edge of the frame.
// Use when computeTemporalHudMask is too slow or the video is very short.
cv::Mat buildMarginMask(int frame_width, int frame_height,
                        float top_frac    = 0.12f,
                        float bottom_frac = 0.12f,
                        float left_frac   = 0.10f,
                        float right_frac  = 0.10f);

// Convert frame to grayscale, apply CLAHE, then zero out HUD regions.
// hud_mask may be empty — if so, the full frame is used.
cv::Mat preprocessFrame(const cv::Mat& frame, const cv::Mat& hud_mask = cv::Mat());

// Preprocess the satellite map image once at startup (no HUD mask).
cv::Mat prepareSatelliteMap(const cv::Mat& satellite_map);

// Gaussian blur → Canny edge detection.
cv::Mat computeEdgeMap(const cv::Mat& gray,
                       double low_thresh  = 50.0,
                       double high_thresh = 150.0);

} // namespace EdgeProcessor
