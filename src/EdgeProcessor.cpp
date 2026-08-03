#include "EdgeProcessor.hpp"
#include <iostream>

namespace EdgeProcessor {

// ---------------------------------------------------------------------------
// computeTemporalHudMask
//
// Kept for API compatibility; returns an empty Mat immediately.
// The temporal-variance approach was too aggressive and removed useful
// terrain pixels.  Use buildMarginMask() instead for a lightweight
// feature-detection mask.
// ---------------------------------------------------------------------------
cv::Mat computeTemporalHudMask(const std::string& /*video_path*/,
                                int   /*n_frames*/,
                                float /*variance_threshold*/)
{
    return cv::Mat();  // caller falls back to buildMarginMask
}

// ---------------------------------------------------------------------------
// buildMarginMask
//
// Creates a feature-detection mask (passed to goodFeaturesToTrack only —
// NOT used to zero display pixels).  For typical drone HUD layouts the
// noise sits in the lower portion of the frame: icons bottom-left, subtitles
// bottom-right.  We exclude only the bottom strip and leave the rest fully
// open so the tracker can use the widest possible terrain area.
// ---------------------------------------------------------------------------
cv::Mat buildMarginMask(int frame_width, int frame_height,
                        float top_frac,
                        float bottom_frac,
                        float left_frac,
                        float right_frac)
{
    cv::Mat mask = cv::Mat::zeros(frame_height, frame_width, CV_8U);
    int x0 = static_cast<int>(frame_width  * left_frac);
    int y0 = static_cast<int>(frame_height * top_frac);
    int x1 = static_cast<int>(frame_width  * (1.0f - right_frac));
    int y1 = static_cast<int>(frame_height * (1.0f - bottom_frac));
    if (x1 > x0 && y1 > y0)
        mask(cv::Rect(x0, y0, x1 - x0, y1 - y0)) = 255;
    return mask;
}

// ---------------------------------------------------------------------------
// preprocessFrame
//
// Converts to greyscale and applies CLAHE for local contrast normalisation.
// Does NOT zero out any pixels — the frame is kept intact for display and
// for optical flow computation.  Noise regions (subtitles, icons) are
// handled by only passing a mask to goodFeaturesToTrack, not by destroying
// pixel data.
// ---------------------------------------------------------------------------
cv::Mat preprocessFrame(const cv::Mat& frame, const cv::Mat& /*hud_mask*/)
{
    cv::Mat gray;
    if (frame.channels() == 1)
        gray = frame.clone();
    else
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // CLAHE: local contrast normalisation — critical for matching satellite
    // imagery (linear radiance) against drone video (gamma-compressed, AGC)
    static thread_local cv::Ptr<cv::CLAHE> clahe =
        cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, gray);

    return gray;
}

// ---------------------------------------------------------------------------
// prepareSatelliteMap
// ---------------------------------------------------------------------------
cv::Mat prepareSatelliteMap(const cv::Mat& satellite_map)
{
    return preprocessFrame(satellite_map, cv::Mat());   // no HUD mask for map
}

// ---------------------------------------------------------------------------
// computeEdgeMap
// ---------------------------------------------------------------------------
cv::Mat computeEdgeMap(const cv::Mat& gray, double low_thresh, double high_thresh)
{
    cv::Mat blurred, edges;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);
    cv::Canny(blurred, edges, low_thresh, high_thresh);
    return edges;
}

} // namespace EdgeProcessor
