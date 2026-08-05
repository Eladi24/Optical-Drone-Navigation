#pragma once
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <utility>
#include <vector>
#include <memory>
#include "DroneSimulation.hpp"

struct PositionEstimate {
    std::pair<double, double> position;
    double confidence;
    int best_match_idx;

    // Optional multi-hypothesis observation for ParticleFilter (see
    // include/ParticleFilter.hpp): (position, inlier-derived score) for each
    // candidate that survived RANSAC, not just the single winner above.
    // Populated by ORBFeatureEstimator only; empty for every other estimator
    // (SIFT/AKAZE/Hybrid/OpticalFlow) and unused by DroneKalmanFilter, which
    // only ever reads position/confidence -- so this is purely additive.
    std::vector<std::pair<std::pair<double, double>, double>> candidates;

    PositionEstimate() : position({0.0, 0.0}), confidence(0.0), best_match_idx(-1) {}
    PositionEstimate(std::pair<double, double> pos, double conf, int idx = -1)
        : position(pos), confidence(conf), best_match_idx(idx) {}
};

// ---------------------------------------------------------------------------
// Per-crop pre-computed feature data.
// Populated once by precompute() and reused every frame.
// ---------------------------------------------------------------------------
struct CachedCropFeatures {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat                   descriptors;
};

class IPositionEstimator {
public:
    virtual ~IPositionEstimator() = default;

    // Call this ONCE before the processing loop to cache crop descriptors.
    virtual void precompute(const std::vector<ReferenceCrop>& /*crops*/) {}

    // Set a per-frame feature-detection mask (255 = use, 0 = exclude).
    // Applied only to the drone frame, not to reference crops.
    // Default no-op — override in feature-based estimators.
    virtual void setFeatureMask(const cv::Mat& /*mask*/) {}

    virtual PositionEstimate estimatePosition(
        const cv::Mat& drone_view,
        const std::vector<ReferenceCrop>& reference_crops,
        const std::pair<double, double>& last_position = {0.0, 0.0}
    ) = 0;

    virtual std::string getName() const = 0;
};

class ORBFeatureEstimator;
class SIFTFeatureEstimator;
class HybridEstimator;
class AKAZEFeatureEstimator;

enum class PositionAlgorithm {
    ORB,
    SIFT,
    HYBRID,
    OPTICAL_FLOW,     // Lucas-Kanade optical flow tracker (Haifa video pipeline)
    AKAZE
};

std::unique_ptr<IPositionEstimator> createPositionEstimator(PositionAlgorithm algorithm);