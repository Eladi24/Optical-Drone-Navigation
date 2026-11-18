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

    // Whether match() may safely be called concurrently, from multiple threads, on this SAME
    // instance (e.g. once per candidate crop within one frame). Default false -- preserves every
    // existing implementation's exact current (sequential, single-threaded) behavior unchanged.
    // Only override to true once the implementation's own internal shared state (e.g. a
    // per-image feature cache) is genuinely synchronized for concurrent access -- see
    // DiskLightGlueMatching for the one implementation that does this, and why it was needed
    // (CPU-only LightGlue inference is too slow per-candidate for a full multi-flight gate
    // without exploiting the real per-frame candidate-level parallelism -- see CLAUDE.md's
    // DISK+LightGlue Investigation Log).
    virtual bool isThreadSafe() const { return false; }

    // Called once per frame, before any match() for that frame (single-threaded). Matchers that
    // cache the QUERY frame's features keyed by cv::Mat::data MUST drop any entry at
    // frame.data here: the per-frame buffer is freed and re-allocated every frame and glibc
    // routinely hands back the same address, so without this, frame N silently reuses frame
    // N-1's (in practice frame 0's) features. Measured 2026-11-17: 146/150 query lookups over 15
    // frames were cache hits, i.e. every DISK+LightGlue / XFeat / dense-DINOv2 split_* run before
    // that date matched candidates against the FIRST frame of the flight. Reference crops live
    // for the whole run, so a live crop can never share the new frame's address.
    virtual void beginFrame(const cv::Mat& /*frame*/) {}
};
