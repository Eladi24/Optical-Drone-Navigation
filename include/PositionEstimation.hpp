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
    // Populated by ORBFeatureEstimator and SIFTFeatureEstimator; empty for
    // every other estimator (AKAZE/Hybrid/OpticalFlow) and unused by
    // DroneKalmanFilter, which only ever reads position/confidence -- so
    // this is purely additive.
    std::vector<std::pair<std::pair<double, double>, double>> candidates;

    // RANSAC inlier count for the winning candidate (0 if no candidate
    // produced a valid homography, or for estimators that don't compute
    // this -- see measurement_valid below to distinguish the two).
    int inliers = 0;

    // False only when every RANSAC-verified candidate was rejected (no
    // geometrically consistent homography found for any candidate above
    // the matching threshold). Distinguishes "no real measurement, filter
    // should predict" from "measured, just genuinely low-confidence" --
    // see STRATEGY.md Sec 4.3 #2 (the RANSAC-failure fallback this
    // replaces used to silently return an unverified guess dressed as a
    // low-confidence one). Defaults true so every estimator that doesn't
    // set it explicitly preserves today's behavior unchanged.
    bool measurement_valid = true;

    PositionEstimate() : position({0.0, 0.0}), confidence(0.0), best_match_idx(-1) {}
    PositionEstimate(std::pair<double, double> pos, double conf, int idx = -1, bool valid = true)
        : position(pos), confidence(conf), best_match_idx(idx), measurement_valid(valid) {}
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

    // Provide the pipeline's geo-referencing scale (metres/pixel, and
    // metres-per-degree for both axes -- the latter differs by cos(lat),
    // see CLAUDE.md's anisotropy findings) so an estimator can convert a
    // homography-refined pixel offset into a real lat/lng position instead
    // of just returning a reference crop's centroid. Call once per run,
    // after the crop grid's scale is known and before the first
    // estimatePosition() call. Default no-op — override in estimators that
    // support continuous (non-quantized) position refinement.
    virtual void setGeoReference(double /*mpp*/,
                                  double /*meters_per_degree_lat*/,
                                  double /*meters_per_degree_lng*/) {}

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
class SplitPipelineEstimator;

enum class PositionAlgorithm {
    ORB,
    SIFT,
    HYBRID,
    OPTICAL_FLOW,     // Lucas-Kanade optical flow tracker (Haifa video pipeline)
    AKAZE,
    SPLIT,            // STRATEGY.md Phase 2: SplitPipelineEstimator, classical
                       // (histogram) retrieval + ORB/RANSAC matching -- see
                       // CLAUDE.md's Phase 2 Investigation Log
    SPLIT_LEARNED_RETRIEVAL, // STRATEGY.md Phase 2: SplitPipelineEstimator,
                       // learned (DeiT-Tiny/ONNX) retrieval + ORB/RANSAC
                       // matching held constant -- isolates the retrieval
                       // variable against SPLIT's already-recorded baseline.
                       // See CLAUDE.md's Phase 2 Investigation Log.
    SPLIT_LEARNED_MATCH_XFEAT, // STRATEGY.md Phase 2: SplitPipelineEstimator,
                       // classical (histogram) retrieval held constant +
                       // learned (XFeat/ONNX) matching -- isolates the
                       // matching variable against SPLIT's baseline, the
                       // complementary comparison to SPLIT_LEARNED_RETRIEVAL
                       // above. See CLAUDE.md's Phase 2 Investigation Log.
    SPLIT_LEARNED,     // STRATEGY.md Phase 2: SplitPipelineEstimator, BOTH
                       // learned stages combined -- DeiT-Tiny/ONNX retrieval
                       // + XFeat/ONNX matching. Not a new variable in
                       // isolation; combines the two already-isolated wins
                       // above to see whether they compound. See CLAUDE.md's
                       // Phase 2 Investigation Log.
    SPLIT_LEARNED_MATCH_DISK_LIGHTGLUE, // STRATEGY.md Phase 2: SplitPipelineEstimator,
                       // classical (histogram) retrieval held constant + learned
                       // (DISK+LightGlue/ONNX) matching -- the second learned-matching
                       // candidate (XFeat was the first), isolates the matching
                       // variable against SPLIT's baseline the same way
                       // SPLIT_LEARNED_MATCH_XFEAT does. See CLAUDE.md's Phase 2
                       // Investigation Log.
    SPLIT_LEARNED_DISK // STRATEGY.md Phase 2: SplitPipelineEstimator, second combined
                       // pipeline -- DeiT-Tiny/ONNX retrieval (SPLIT_LEARNED_RETRIEVAL's
                       // winning stage) + DISK+LightGlue/ONNX matching
                       // (SPLIT_LEARNED_MATCH_DISK_LIGHTGLUE's winning stage), mirroring
                       // SPLIT_LEARNED's own DeiT+XFeat precedent but with the
                       // 0%-invalid-rate matcher instead. Not a new isolated variable;
                       // combines two already-individually-measured wins to see whether
                       // they compound the way SPLIT_LEARNED's did. See CLAUDE.md's
                       // Phase 2 Investigation Log.
};

std::unique_ptr<IPositionEstimator> createPositionEstimator(PositionAlgorithm algorithm);