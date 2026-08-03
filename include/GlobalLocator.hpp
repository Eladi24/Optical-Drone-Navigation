#pragma once

#include <opencv2/opencv.hpp>
#include <utility>
#include <string>

struct InitializationData {
    bool success;
    int frame_index;
    std::pair<double, double> coordinates;
};

class GlobalLocator {
public:
    // -----------------------------------------------------------------------
    // Multi-scale edge-based initialization.
    //
    // Searches the first ~12 s of the video at 7 scale hypotheses in
    // parallel.  Uses Canny edge template matching on CLAHE-preprocessed
    // frames, which is robust to low-texture coastal/harbor scenes.
    //
    // @param video_path    Path to the drone video.
    // @param satellite_map Satellite map image (BGR or greyscale).
    // @param center_*      Map centre geo-coordinates and pixel location.
    // @param mpp           Map metres-per-pixel.
    // @param m_per_deg_*   Degrees-to-metres conversion for this latitude.
    // @param frame_skip    Evaluate every Nth frame (default 10).
    // @param feature_mask  Optional mask (255=use, 0=exclude) applied when
    //                      computing drone-frame edge maps. Pass an empty Mat
    //                      to disable masking.
    // -----------------------------------------------------------------------
    static InitializationData findStartingPosition(
        const std::string& video_path,
        const cv::Mat&     satellite_map,
        double center_lat,  double center_lng,
        int    center_x,    int    center_y,
        double mpp,
        double m_per_deg_lat,
        double m_per_deg_lng,
        int    frame_skip    = 10,
        const cv::Mat& feature_mask = cv::Mat()
    );
};
