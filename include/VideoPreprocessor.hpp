#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "CameraModel.hpp"

struct VideoMetadata {
    double fps;
    int total_frames;
    int width;
    int height;
    double estimated_mean_altitude; // metres; -1 if unknown
    double estimated_mean_angle;    // pitch angle in degrees; -1 if nadir/unknown
    std::string processed_video_path;
    // Feature-detection mask: 255 = valid terrain, 0 = exclude (logos/HUD/subtitles).
    // Pass this to IPositionEstimator::setFeatureMask() and to OpticalFlowTracker.
    cv::Mat feature_mask;
};

class VideoPreprocessor {
public:
    // masks_dir: directory where per-video mask PNGs are saved and reloaded.
    //   If the mask PNG already exists the interactive editor is skipped.
    VideoPreprocessor(const std::string& input_video_path,
                      const std::string& output_video_path,
                      const std::string& masks_dir = "masks");

    // Run the full preprocessing pipeline:
    //   1. Auto-detect static overlays (temporal variance)
    //   2. Interactive mask refinement (unless cached mask PNG exists)
    //   3. Estimate camera angle from horizon detection
    //   4. Estimate altitude interactively (user draws line over known object)
    //   5. Write CLAHE-enhanced output video
    VideoMetadata process(int frame_skip = 1);

private:
    std::string input_path_;
    std::string output_path_;
    std::string masks_dir_;

    // 1. Temporal variance across sample_frames evenly-spaced frames.
    //    Returns a mask where 0 = static HUD/logo, 255 = moving terrain.
    cv::Mat generateStaticOverlayMask(cv::VideoCapture& cap, int sample_frames = 60);

    // 2. Interactive OpenCV window for mask refinement.
    //    Shows auto-detected mask overlaid on first frame.
    //    Left-drag: add exclusion rectangle. Right-drag: erase exclusion.
    //    'S' = save & continue, 'R' = reset to auto_mask.
    //    Saves the result to masks_dir_/<video_stem>_mask.png for reuse.
    cv::Mat interactiveRefineOverlayMask(const cv::Mat& first_frame,
                                         const cv::Mat& auto_mask,
                                         const std::string& mask_path);

    // 3. Horizon detection → pitch angle in degrees; returns -1 if nadir.
    double estimateCameraAngle(const cv::Mat& frame);

    // 4. Interactive frame selection (trackbar / a-d / A-D stepping, same
    //    controls as the GroundTruth tool) followed by a line-draw over a
    //    known object -> altitude in metres. Frame 0 isn't always usable
    //    (e.g. Sample 1 opens over open water) so the user picks the frame.
    //    Assumes camera_hfov_deg horizontal field of view.
    double estimateAltitudeFromReference(cv::VideoCapture& cap,
                                          int total_frames,
                                          double camera_hfov_deg = CameraModel::kDefaultHfovDeg);

    // Camera calibration (identity until proper calibration data is loaded)
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
};