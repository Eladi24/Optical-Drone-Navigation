#include "TemplateMatchingEstimator.hpp"
#include <opencv2/opencv.hpp>

PositionEstimate TemplateMatchingEstimator::estimatePosition(
    const cv::Mat& drone_view,
    const std::vector<ReferenceCrop>& reference_crops,
    const std::pair<double, double>& last_position)
{
    if (drone_view.empty() || reference_crops.empty()) {
        return PositionEstimate(last_position, 0.0, -1);
    }

    double best_score = -1.0;
    int best_idx = -1;
    std::pair<double, double> best_position = last_position;

    for (size_t idx = 0; idx < reference_crops.size(); idx++) {
        const auto& crop = reference_crops[idx];
        
        if (crop.image.empty()) continue; 

        cv::Mat comparison_image;
        if (drone_view.size() != crop.image.size()) {
            cv::resize(crop.image, comparison_image, drone_view.size());
        } else {
            comparison_image = crop.image;
        }

        cv::Mat result;
        cv::matchTemplate(comparison_image, drone_view, result, cv::TM_CCOEFF_NORMED);

        double min_val, max_val;
        cv::minMaxLoc(result, &min_val, &max_val);

        if (max_val > best_score) {
            best_score = max_val;
            best_idx = static_cast<int>(idx);
            best_position = crop.coordinates; 
        }
    }

    double confidence = std::max(0.0, std::min(1.0, best_score));
    return PositionEstimate(best_position, confidence, best_idx);
}