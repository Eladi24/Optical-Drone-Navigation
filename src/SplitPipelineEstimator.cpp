#include "SplitPipelineEstimator.hpp"
#include "CoordinateUtils.hpp"
#include "VideoProcessing.hpp"  // kReferenceCropMeters
#include <opencv2/calib3d.hpp>
#include <algorithm>

SplitPipelineEstimator::SplitPipelineEstimator(std::unique_ptr<IRetrievalStage> retrieval,
                                                std::unique_ptr<IMatchingStage> matching,
                                                std::string name,
                                                int top_k)
    : retrieval_(std::move(retrieval))
    , matching_(std::move(matching))
    , name_(std::move(name))
    , top_k_(top_k)
{
}

void SplitPipelineEstimator::precompute(const std::vector<ReferenceCrop>& crops)
{
    retrieval_->buildIndex(crops);
}

void SplitPipelineEstimator::setFeatureMask(const cv::Mat& mask)
{
    retrieval_->setFeatureMask(mask);
    matching_->setFeatureMask(mask);
}

void SplitPipelineEstimator::setGeoReference(double mpp, double meters_per_degree_lat,
                                              double meters_per_degree_lng)
{
    mpp_ = mpp;
    meters_per_degree_lat_ = meters_per_degree_lat;
    meters_per_degree_lng_ = meters_per_degree_lng;
}

PositionEstimate SplitPipelineEstimator::estimatePosition(
    const cv::Mat& drone_view,
    const std::vector<ReferenceCrop>& reference_crops,
    const std::pair<double, double>& last_position)
{
    if (drone_view.empty() || reference_crops.empty()) {
        PositionEstimate result(last_position, 0.0, -1);
        result.measurement_valid = false;
        return result;
    }

    std::vector<int> candidate_indices = retrieval_->retrieve(drone_view, top_k_);
    if (candidate_indices.empty()) {
        PositionEstimate result(last_position, 0.3, -1);
        result.measurement_valid = false;
        return result;
    }

    int best_idx     = -1;
    int best_inliers = 0;
    cv::Mat best_H;
    // Every candidate the matching stage verified, not just the winner --
    // feeds recall@k instrumentation (VideoProcessing.cpp) and
    // ParticleFilter's multi-hypothesis update, same as ORBFeatureEstimator.
    std::vector<std::pair<std::pair<double, double>, double>> surviving_candidates;

    for (int idx : candidate_indices) {
        if (idx < 0 || idx >= static_cast<int>(reference_crops.size())) continue;

        MatchResult m = matching_->match(drone_view, reference_crops[idx]);
        if (m.homography.empty() || m.inliers == 0) continue;

        surviving_candidates.push_back({reference_crops[idx].coordinates,
                                         static_cast<double>(m.inliers)});

        if (m.inliers > best_inliers) {
            best_inliers = m.inliers;
            best_idx     = idx;
            best_H       = m.homography;
        }
    }

    // No candidate produced a geometrically consistent homography -- no real
    // measurement this frame. Same honest-reporting contract as
    // ORBFeatureEstimator (STRATEGY.md Sec 4.3 #2): let the Kalman filter
    // predict instead of returning a known-bad guess dressed as low-confidence.
    if (best_idx == -1) {
        PositionEstimate result(last_position, 0.0, -1);
        result.measurement_valid = false;
        return result;
    }

    // Continuous position refinement: project the drone frame's center
    // through the winning homography instead of returning the matched
    // crop's centroid (STRATEGY.md Sec 4.2) -- identical logic to
    // ORBFeatureEstimator::estimatePosition().
    std::pair<double, double> refined_position = reference_crops[best_idx].coordinates;
    if (mpp_ > 0.0) {
        const cv::Mat& crop_image = reference_crops[best_idx].image;
        std::vector<cv::Point2f> src{cv::Point2f(drone_view.cols / 2.0f, drone_view.rows / 2.0f)};
        std::vector<cv::Point2f> dst;
        cv::perspectiveTransform(src, dst, best_H);

        double offset_px_x = std::abs(dst[0].x - crop_image.cols / 2.0);
        double offset_px_y = std::abs(dst[0].y - crop_image.rows / 2.0);
        double offset_m_x  = offset_px_x * mpp_;
        double offset_m_y  = offset_px_y * mpp_;

        if (offset_m_x <= kReferenceCropMeters * 0.75 && offset_m_y <= kReferenceCropMeters * 0.75) {
            refined_position = CoordinateUtils::pixelToLatLng(
                static_cast<int>(dst[0].x), static_cast<int>(dst[0].y),
                reference_crops[best_idx].coordinates.first,
                reference_crops[best_idx].coordinates.second,
                crop_image.cols / 2, crop_image.rows / 2,
                mpp_, meters_per_degree_lat_, meters_per_degree_lng_);
        }
    }

    const double confidence = std::min(1.0, 0.3 + (best_inliers / 30.0) * 0.7);

    PositionEstimate result(refined_position, confidence, best_idx);
    result.candidates = std::move(surviving_candidates);
    result.inliers     = best_inliers;
    return result;
}
