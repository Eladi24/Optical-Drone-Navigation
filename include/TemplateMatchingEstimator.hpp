#pragma once
#include "PositionEstimation.hpp"

class TemplateMatchingEstimator : public IPositionEstimator {
public:
    TemplateMatchingEstimator() = default;
    
    PositionEstimate estimatePosition(
        const cv::Mat& drone_view,
        const std::vector<std::pair<double, double>>& ref_crop_coords,
        const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash>& reference_crops,
        const std::pair<double, double>& last_position = {0.0, 0.0}
    ) override;
    
    std::string getName() const override { return "template"; }
};