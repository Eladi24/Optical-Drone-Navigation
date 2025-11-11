#pragma once
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <utility>
#include <vector>
#include <memory>
#include "DroneSimulation.hpp"

// Result structure for position estimation
struct PositionEstimate {
    std::pair<double, double> position;
    double confidence;
    int best_match_idx;
    
    PositionEstimate() : position({0.0, 0.0}), confidence(0.0), best_match_idx(-1) {}
    PositionEstimate(std::pair<double, double> pos, double conf, int idx = -1)
        : position(pos), confidence(conf), best_match_idx(idx) {}
};

// Strategy interface (abstract base class)
class IPositionEstimator {
public:
    virtual ~IPositionEstimator() = default;
    
    // Main estimation method
    virtual PositionEstimate estimatePosition(
        const cv::Mat& drone_view,
        const std::vector<ReferenceCrop>& reference_crops,
        const std::pair<double, double>& last_position = {0.0, 0.0}
    ) = 0;
    
    // Get algorithm name
    virtual std::string getName() const = 0;
};

// Forward declare estimator classes
class TemplateMatchingEstimator;
class ORBFeatureEstimator;
class SIFTFeatureEstimator;
class SURFFeatureEstimator;
class HybridEstimator;
class SmoothedEstimator;

// Enum for algorithm selection
enum class PositionAlgorithm {
    TEMPLATE,
    ORB,
    SIFT,
    SURF,
    HYBRID,
    SMOOTHED
};

// Factory function to create estimators
std::unique_ptr<IPositionEstimator> createPositionEstimator(PositionAlgorithm algorithm);