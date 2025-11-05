#include "TemplateMatchingEstimator.hpp"

PositionEstimate TemplateMatchingEstimator::estimatePosition(
    const cv::Mat& drone_view,
    const std::vector<std::pair<double, double>>& ref_crop_coords,
    const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash>& reference_crops,
    const std::pair<double, double>& last_position)
{
    if (drone_view.empty() || ref_crop_coords.empty()) {
        return PositionEstimate(last_position, 0.0, -1);
    }

    double best_score = -1.0;
    int best_idx = -1;
    std::pair<double, double> best_position = last_position;

    // Iterate through all reference crops
    for (size_t idx = 0; idx < ref_crop_coords.size(); idx++) {
        const auto& coords = ref_crop_coords[idx];
        
        // Check if the coordinates exist in the map
        auto it = reference_crops.find(coords);
        if (it == reference_crops.end()) {
            continue;
        }
        
        const cv::Mat& ref_crop = it->second;
        
        if (ref_crop.empty()) {
            continue;
        }

        // Ensure comparable sizes - resize reference crop if needed
        cv::Mat comparison_image;
        if (drone_view.size() != ref_crop.size()) {
            cv::resize(ref_crop, comparison_image, drone_view.size());
        } else {
            comparison_image = ref_crop;
        }

        // Perform template matching
        cv::Mat result;
        cv::matchTemplate(comparison_image, drone_view, result, cv::TM_CCOEFF_NORMED);

        // Get the best match score
        double min_val, max_val;
        cv::minMaxLoc(result, &min_val, &max_val);

        // Update best match if this is better
        if (max_val > best_score) {
            best_score = max_val;
            best_idx = static_cast<int>(idx);
            best_position = coords;
        }
    }

    // Ensure confidence is in valid range [0, 1]
    double confidence = std::max(0.0, std::min(1.0, best_score));

    return PositionEstimate(best_position, confidence, best_idx);
}