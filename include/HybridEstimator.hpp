#pragma once
#include "PositionEstimation.hpp"
#include "TemplateMatchingEstimator.hpp"
#include "ORBFeatureEstimator.hpp"

class HybridEstimator : public IPositionEstimator {
public:
    HybridEstimator();
    
    PositionEstimate estimatePosition(
        const cv::Mat& drone_view,
        const std::vector<ReferenceCrop>& reference_crops,
        const std::pair<double, double>& last_position = {0.0, 0.0}
    ) override;
    
    std::string getName() const override { return "hybrid"; }

private:
    TemplateMatchingEstimator template_estimator_;
    ORBFeatureEstimator orb_estimator_;
};