#include "SmoothedEstimator.hpp"
#include <algorithm>

SmoothedEstimator::SmoothedEstimator(int top_matches)
    : top_matches_(top_matches)
{
}

PositionEstimate SmoothedEstimator::estimatePosition(
    const cv::Mat& drone_view,
    const std::vector<std::pair<double, double>>& ref_crop_coords,
    const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash>& reference_crops,
    const std::pair<double, double>& last_position)
{
    if (drone_view.empty() || ref_crop_coords.empty()) {
        return PositionEstimate(last_position, 0.0, -1);
    }

    // Get all matches with scores
    std::vector<std::tuple<int, double, std::pair<double, double>>> all_matches;

    // Calculate match score for each reference crop
    for (size_t idx = 0; idx < ref_crop_coords.size(); idx++) {
        const auto& coords = ref_crop_coords[idx];
        
        auto it = reference_crops.find(coords);
        if (it == reference_crops.end()) {
            continue;
        }
        
        const cv::Mat& ref_crop = it->second;
        
        if (ref_crop.empty()) {
            continue;
        }

        // Ensure comparable sizes
        cv::Mat comparison_image;
        if (drone_view.size() != ref_crop.size()) {
            cv::resize(ref_crop, comparison_image, drone_view.size());
        } else {
            comparison_image = ref_crop;
        }

        // Calculate match score using template matching
        cv::Mat result;
        cv::matchTemplate(comparison_image, drone_view, result, cv::TM_CCOEFF_NORMED);

        double score;
        cv::minMaxLoc(result, nullptr, &score);
        
        all_matches.push_back(std::make_tuple(static_cast<int>(idx), score, coords));
    }

    // Sort by score in descending order
    std::sort(all_matches.begin(), all_matches.end(),
              [](const auto& a, const auto& b) {
                  return std::get<1>(a) > std::get<1>(b);
              });

    // Use weighted average of top N matches
    int matches_to_use = std::min(top_matches_, static_cast<int>(all_matches.size()));
    
    if (matches_to_use == 0) {
        return PositionEstimate(last_position, 0.0, -1);
    }

    double total_weight = 0.0;
    double weighted_lat = 0.0;
    double weighted_lng = 0.0;
    double max_confidence = 0.0;
    int best_idx = -1;

    for (int i = 0; i < matches_to_use; i++) {
        const auto& match = all_matches[i];
        double score = std::get<1>(match);
        const auto& coords = std::get<2>(match);
        int idx = std::get<0>(match);

        if (i == 0) {
            max_confidence = score;
            best_idx = idx;
        }

        // Apply score^2 weighting to emphasize better matches
        double weight = score * score;
        total_weight += weight;
        weighted_lat += coords.first * weight;
        weighted_lng += coords.second * weight;
    }

    if (total_weight > 0.0) {
        std::pair<double, double> smoothed_position = {
            weighted_lat / total_weight,
            weighted_lng / total_weight
        };
        
        // Confidence is slightly lower than max due to averaging
        double confidence = std::max(0.0, std::min(1.0, max_confidence * 0.9));
        
        return PositionEstimate(smoothed_position, confidence, best_idx);
    }

    return PositionEstimate(last_position, 0.0, -1);
}