#include "SURFFeatureEstimator.hpp"
#include <algorithm>

SURFFeatureEstimator::SURFFeatureEstimator(double hessian_threshold, int num_octaves,
                                           int num_octave_layers, bool extended)
    : surf_detector_(cv::xfeatures2d::SURF::create(hessian_threshold, num_octaves,
                                                    num_octave_layers, extended))
{
}

PositionEstimate SURFFeatureEstimator::estimatePosition(
    const cv::Mat& drone_view,
    const std::vector<ReferenceCrop>& reference_crops,
    const std::pair<double, double>& last_position)
{
    if (drone_view.empty() || reference_crops.empty()) {
        return PositionEstimate(last_position, 0.0, -1);
    }

    // Extract keypoints and descriptors from drone view
    std::vector<cv::KeyPoint> keypoints_drone;
    cv::Mat descriptors_drone;
    surf_detector_->detectAndCompute(drone_view, cv::noArray(), keypoints_drone, descriptors_drone);

    if (keypoints_drone.size() < 10 || descriptors_drone.empty()) {
        return PositionEstimate(last_position, 0.3, -1);
    }

    // Match against each reference crop
    std::vector<std::pair<int, float>> match_scores;
    cv::BFMatcher matcher(cv::NORM_L2);

    for (size_t idx = 0; idx < reference_crops.size(); idx++) {
        const auto& crop = reference_crops[idx];
        
        if (crop.image.empty()) {
            continue;
        }

        // Extract features from reference crop
        std::vector<cv::KeyPoint> keypoints_ref;
        cv::Mat descriptors_ref;
        surf_detector_->detectAndCompute(crop.image, cv::noArray(), keypoints_ref, descriptors_ref);

        if (keypoints_ref.size() < 10 || descriptors_ref.empty()) {
            continue;
        }

        // Match descriptors using KNN
        std::vector<std::vector<cv::DMatch>> knn_matches;
        try {
            matcher.knnMatch(descriptors_drone, descriptors_ref, knn_matches, 2);
        } catch (const cv::Exception& e) {
            continue;
        }

        // Filter good matches using Lowe's ratio test
        std::vector<cv::DMatch> good_matches;
        for (const auto& match_pair : knn_matches) {
            if (match_pair.size() >= 2) {
                if (match_pair[0].distance < 0.75f * match_pair[1].distance) {
                    good_matches.push_back(match_pair[0]);
                }
            }
        }

        // Calculate score
        float match_score = 0.0f;
        if (!good_matches.empty()) {
            float avg_distance = 0.0f;
            for (const auto& m : good_matches) {
                avg_distance += m.distance;
            }
            avg_distance /= good_matches.size();

            match_score = static_cast<float>(good_matches.size()) / (avg_distance / 100.0f + 1.0f);
        }

        match_scores.push_back({static_cast<int>(idx), match_score});
    }

    // Sort by score
    std::sort(match_scores.begin(), match_scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    if (match_scores.empty()) {
        return PositionEstimate(last_position, 0.3, -1);
    }

    int best_idx = match_scores[0].first;
    std::pair<double, double> best_position = reference_crops[best_idx].coordinates;
    
    // Normalize confidence
    double confidence = std::min(1.0, match_scores[0].second / 100.0);
    confidence = std::max(0.3, confidence);

    return PositionEstimate(best_position, confidence, best_idx);
}