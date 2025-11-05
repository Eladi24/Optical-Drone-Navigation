#pragma once
#include "PositionEstimation.hpp"
#include <opencv2/xfeatures2d.hpp>

class SIFTFeatureEstimator : public IPositionEstimator {
public:
    SIFTFeatureEstimator(int num_features = 0, int num_octave_layers = 3, 
                         double contrast_threshold = 0.04, double edge_threshold = 10.0);
    
    PositionEstimate estimatePosition(
        const cv::Mat& drone_view,
        const std::vector<std::pair<double, double>>& ref_crop_coords,
        const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash>& reference_crops,
        const std::pair<double, double>& last_position = {0.0, 0.0}
    ) override;
    
    std::string getName() const override { return "sift"; }

private:
    cv::Ptr<cv::SIFT> sift_detector_;
};