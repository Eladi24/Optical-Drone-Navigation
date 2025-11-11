#pragma once
#include "PositionEstimation.hpp"

class ORBFeatureEstimator : public IPositionEstimator {
public:
    ORBFeatureEstimator(int num_features = 500);
    
    PositionEstimate estimatePosition(
        const cv::Mat& drone_view,
        const std::vector<ReferenceCrop>& reference_crops,
        const std::pair<double, double>& last_position = {0.0, 0.0}
    ) override;
    
    std::string getName() const override { return "orb"; }

private:
    cv::Ptr<cv::ORB> orb_detector_;
};