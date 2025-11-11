#include "ORBFeatureEstimator.hpp"

ORBFeatureEstimator::ORBFeatureEstimator(int num_features)
    : orb_detector_(cv::ORB::create(num_features))
{
}

PositionEstimate ORBFeatureEstimator::estimatePosition(
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
    orb_detector_->detectAndCompute(drone_view, cv::noArray(), keypoints_drone, descriptors_drone);

    if (keypoints_drone.size() < 10 || descriptors_drone.empty()) {
        // Not enough features in drone view, return low confidence
        return PositionEstimate(last_position, 0.3, -1);
    }

    // Match against each reference crop
    std::vector<std::pair<int, float>> match_scores;
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    for (size_t idx = 0; idx < reference_crops.size(); idx++) {
        const auto& crop = reference_crops[idx];
        if (crop.image.empty()) {
            continue;
        }

        // Extract features from reference crop
        std::vector<cv::KeyPoint> keypoints_ref;
        cv::Mat descriptors_ref;
        orb_detector_->detectAndCompute(crop.image, cv::noArray(), keypoints_ref, descriptors_ref);

        if (keypoints_ref.size() < 10 || descriptors_ref.empty()) {
            continue;
        }

        // Match descriptors using KNN
        std::vector<std::vector<cv::DMatch>> knn_matches;
        try {
            matcher.knnMatch(descriptors_drone, descriptors_ref, knn_matches, 2);
        } catch (const cv::Exception& e) {
            // Matching failed, skip this reference crop
            continue;
        }

        // Filter good matches using ratio test (Lowe's ratio test)
        std::vector<cv::DMatch> good_matches;
        for (const auto& match_pair : knn_matches) {
            if (match_pair.size() >= 2) {
                if (match_pair[0].distance < 0.75f * match_pair[1].distance) {
                    good_matches.push_back(match_pair[0]);
                }
            }
        }

        // Calculate score based on number and quality of matches
        float match_score = 0.0f;
        if (!good_matches.empty()) {
            // Score combines number of good matches and their quality
            float avg_distance = 0.0f;
            for (const auto& m : good_matches) {
                avg_distance += m.distance;
            }
            avg_distance /= good_matches.size();

            // Higher score for more matches and lower distance
            // Normalize by dividing by max possible distance (256 for Hamming)
            match_score = static_cast<float>(good_matches.size()) / (avg_distance / 256.0f + 1.0f);
        }

        match_scores.push_back({static_cast<int>(idx), match_score});
    }

    // Sort by score in descending order
    std::sort(match_scores.begin(), match_scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Return best match or last position if no matches
    if (match_scores.empty()) {
        return PositionEstimate(last_position, 0.3, -1);
    }

    int best_idx = match_scores[0].first;
    std::pair<double, double> best_position = reference_crops[best_idx].coordinates;
    // Normalize confidence to [0, 1] range (empirically tune the divisor)
    double confidence = std::min(1.0, match_scores[0].second / 50.0);
    confidence = std::max(0.3, confidence); // Minimum confidence of 0.3

    return PositionEstimate(best_position, confidence, best_idx);
}