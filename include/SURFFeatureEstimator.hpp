#pragma once
#include "PositionEstimation.hpp"
#include <opencv2/xfeatures2d.hpp>

class SURFFeatureEstimator : public IPositionEstimator {
public:
    SURFFeatureEstimator(double hessian_threshold = 400.0, int num_octaves = 4,
                         int num_octave_layers = 3, bool extended = false);
    
    PositionEstimate estimatePosition(
        const cv::Mat& drone_view,
        const std::vector<std::pair<double, double>>& ref_crop_coords,
        const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash>& reference_crops,
        const std::pair<double, double>& last_position = {0.0, 0.0}
    ) override;
    
    std::string getName() const override { return "surf"; }

private:
    cv::Ptr<cv::xfeatures2d::SURF> surf_detector_;
};