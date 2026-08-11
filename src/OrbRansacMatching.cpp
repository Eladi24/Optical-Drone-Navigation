#include "OrbRansacMatching.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

OrbRansacMatching::OrbRansacMatching(int num_features)
    : orb_detector_(cv::ORB::create(num_features))
{
}

const OrbRansacMatching::CachedFeatures*
OrbRansacMatching::getCachedFeatures(const cv::Mat& image) const
{
    const uchar* key = image.data;
    auto it = feature_cache_.find(key);
    if (it != feature_cache_.end())
        return it->second.valid ? &it->second : nullptr;

    CachedFeatures entry;
    orb_detector_->detectAndCompute(image, cv::noArray(), entry.keypoints, entry.descriptors);
    // Same threshold and rationale as ORBFeatureEstimator.cpp's
    // MIN_VALID_KEYPOINTS -- see CLAUDE.md's keypoint-count-distribution
    // findings for why 150, not a lower or per-descriptor-count value.
    static constexpr size_t MIN_VALID_KEYPOINTS = 150;
    entry.valid = !entry.descriptors.empty() &&
                  entry.keypoints.size() >= MIN_VALID_KEYPOINTS;

    auto [ins, _] = feature_cache_.emplace(key, std::move(entry));
    return ins->second.valid ? &ins->second : nullptr;
}

MatchResult OrbRansacMatching::match(const cv::Mat& frame, const ReferenceCrop& crop)
{
    MatchResult result;
    if (frame.empty() || crop.image.empty()) return result;

    std::vector<cv::KeyPoint> kp_frame;
    cv::Mat desc_frame;
    cv::InputArray mask_input = feature_mask_.empty() ? cv::noArray() : cv::InputArray(feature_mask_);
    orb_detector_->detectAndCompute(frame, mask_input, kp_frame, desc_frame);
    if (kp_frame.size() < 10 || desc_frame.empty()) return result;

    const CachedFeatures* ref = getCachedFeatures(crop.image);
    if (!ref) return result;  // featureless or invalid crop

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    try {
        matcher.knnMatch(desc_frame, ref->descriptors, knn, 2);
    } catch (...) {
        return result;
    }

    // Lowe's ratio test -- 0.75 is the standard threshold for ORB.
    std::vector<cv::DMatch> good;
    good.reserve(knn.size());
    for (const auto& m : knn)
        if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance)
            good.push_back(m[0]);

    // Homography requires at least 8 well-distributed correspondences.
    if (static_cast<int>(good.size()) < 8) return result;

    std::vector<cv::Point2f> src_pts, dst_pts;
    src_pts.reserve(good.size());
    dst_pts.reserve(good.size());
    for (const auto& m : good) {
        src_pts.push_back(kp_frame[m.queryIdx].pt);
        dst_pts.push_back(ref->keypoints[m.trainIdx].pt);
    }

    cv::Mat mask;
    cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 5.0, mask);
    if (H.empty()) return result;

    int inliers = cv::countNonZero(mask);
    if (inliers == 0) return result;

    result.homography = H;
    result.inliers = inliers;
    return result;
}
