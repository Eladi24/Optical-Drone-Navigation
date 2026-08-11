#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "DroneSimulation.hpp"  // ReferenceCrop

// STRATEGY.md Phase 2: split retrieval (find candidate crops) from matching
// (verify one crop) into independently swappable, independently measurable
// stages -- see CLAUDE.md's Phase 2 Investigation Log for the diagnosis
// this responds to (Phase 0: retrieval, not matching or fusion, is the
// dominant, measurable bottleneck).

// frame -> ranked list of candidate reference-crop indices, best first.
class IRetrievalStage {
public:
    virtual ~IRetrievalStage() = default;

    // Called once before the video loop (mirrors
    // IPositionEstimator::precompute()'s existing contract).
    virtual void buildIndex(const std::vector<ReferenceCrop>& crops) = 0;

    // Per-frame feature-detection mask (255=use, 0=exclude), applied only to
    // the query frame -- same semantics as IPositionEstimator::setFeatureMask().
    // Default no-op -- override in implementations that can honor it.
    virtual void setFeatureMask(const cv::Mat& /*mask*/) {}

    virtual std::vector<int> retrieve(const cv::Mat& frame, int k) = 0;
};

// One (frame, candidate crop) pair -> verified relative pose + inlier count.
struct MatchResult {
    cv::Mat homography;   // empty if the match failed
    int inliers = 0;
};

class IMatchingStage {
public:
    virtual ~IMatchingStage() = default;

    // Same semantics/default as IRetrievalStage::setFeatureMask() above.
    virtual void setFeatureMask(const cv::Mat& /*mask*/) {}

    virtual MatchResult match(const cv::Mat& frame, const ReferenceCrop& crop) = 0;
};
