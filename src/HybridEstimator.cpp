#include "HybridEstimator.hpp"
#include "CoordinateUtils.hpp"
#include <cmath>

HybridEstimator::HybridEstimator(double agreement_threshold_m)
    : orb_(500), sift_(500), agreement_threshold_m_(agreement_threshold_m)
{}

void HybridEstimator::precompute(const std::vector<ReferenceCrop>& crops)
{
    orb_.precomputeAll(crops);
    sift_.precompute(crops);
}

void HybridEstimator::setFeatureMask(const cv::Mat& mask)
{
    orb_.setFeatureMask(mask);
    sift_.setFeatureMask(mask);
}

PositionEstimate HybridEstimator::estimatePosition(
    const cv::Mat&                    drone_view,
    const std::vector<ReferenceCrop>& reference_crops,
    const std::pair<double, double>&  last_position)
{
    if (drone_view.empty() || reference_crops.empty())
        return PositionEstimate(last_position, 0.0, -1);

    PositionEstimate orb_est  = orb_.estimatePosition(drone_view, reference_crops, last_position);
    PositionEstimate sift_est = sift_.estimatePosition(drone_view, reference_crops, last_position);

    const bool orb_ok  = orb_est.confidence  > 0.50;
    const bool sift_ok = sift_est.confidence > 0.45;

    if (orb_ok && sift_ok) {
        // Measure geographic distance between the two estimates.
        // Use a fast flat-earth approximation — good enough within a few km.
        const double lat_diff_m = (orb_est.position.first  - sift_est.position.first)  * 111320.0;
        const double lng_mid    = (orb_est.position.first   + sift_est.position.first)  * 0.5 * M_PI / 180.0;
        const double lng_diff_m = (orb_est.position.second - sift_est.position.second) * 111320.0 * std::cos(lng_mid);
        const double dist_m     = std::sqrt(lat_diff_m * lat_diff_m + lng_diff_m * lng_diff_m);

        if (dist_m <= agreement_threshold_m_) {
            // Both agree → weighted average, boosted confidence
            const double w_sift = 0.6, w_orb = 0.4;
            double lat = w_sift * sift_est.position.first  + w_orb * orb_est.position.first;
            double lng = w_sift * sift_est.position.second + w_orb * orb_est.position.second;
            double conf = std::min(0.98, 0.5 * (sift_est.confidence + orb_est.confidence) + 0.15);
            return PositionEstimate({lat, lng}, conf, sift_est.best_match_idx);
        }

        // Disagree → trust SIFT (more robust descriptors)
        return sift_est;
    }

    if (sift_ok) return sift_est;
    if (orb_ok)  return orb_est;

    // Both failed → coast on last known position with very low confidence
    return PositionEstimate(last_position, 0.1, -1);
}
