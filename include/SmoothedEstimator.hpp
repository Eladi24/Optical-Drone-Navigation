#pragma once
#include "PositionEstimation.hpp"

class SmoothedEstimator : public IPositionEstimator {
public:
    SmoothedEstimator(int top_matches = 3);
    
    PositionEstimate estimatePosition(
        const cv::Mat& drone_view,
        const std::vector<std::pair<double, double>>& ref_crop_coords,
        const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash>& reference_crops,
        const std::pair<double, double>& last_position = {0.0, 0.0}
    ) override;
    
    std::string getName() const override { return "smoothed"; }

private:
    int top_matches_;
};