#pragma once
#include "PositionEstimation.hpp"
#include "ORBFeatureEstimator.hpp"
#include "SIFTFeatureEstimator.hpp"
#include <memory>

// ---------------------------------------------------------------------------
// HybridEstimator — ORB + SIFT consensus position estimator.
//
// Decision logic per frame:
//   1. Run both ORB and SIFT on the drone frame.
//   2. If both succeed AND agree within agreement_threshold_m → weighted
//      average (SIFT 0.6, ORB 0.4) with boosted confidence.
//   3. If only one exceeds its confidence threshold → use that result.
//   4. If both fail (featureless water/haze) → return last_position with
//      low confidence so the Kalman filter coasts on its motion model.
// ---------------------------------------------------------------------------
class HybridEstimator : public IPositionEstimator {
public:
    // agreement_threshold_m: max geographic distance (metres) at which two
    //   estimates are considered "in agreement".
    explicit HybridEstimator(double agreement_threshold_m = 30.0);

    void precompute(const std::vector<ReferenceCrop>& crops) override;

    void setFeatureMask(const cv::Mat& mask) override;

    PositionEstimate estimatePosition(
        const cv::Mat&                    drone_view,
        const std::vector<ReferenceCrop>& reference_crops,
        const std::pair<double, double>&  last_position = {0.0, 0.0}
    ) override;

    std::string getName() const override { return "hybrid"; }

private:
    ORBFeatureEstimator  orb_;
    SIFTFeatureEstimator sift_;
    double               agreement_threshold_m_;
};
