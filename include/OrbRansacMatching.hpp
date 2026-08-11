#pragma once
#include "RetrievalStage.hpp"
#include <unordered_map>

// Matching stage: given one candidate crop, independently detects ORB
// features in the query frame and verifies a RANSAC homography against it.
// Lifted from ORBFeatureEstimator's Stage 1 (Lowe's ratio test) + Stage 2
// (RANSAC) -- see CLAUDE.md's Phase 2 Investigation Log for why matching
// must be self-sufficient per candidate rather than reuse correspondences a
// retrieval stage happened to compute internally: a learned (global
// embedding) retrieval stage produces a ranked crop list and nothing else,
// so matching can't assume pre-computed correspondences exist.
//
// Unlike ORBFeatureEstimator, this runs sequentially (called once per
// retrieved candidate, not once per crop in the whole database), so its
// descriptor cache needs no locking -- SplitPipelineEstimator never calls
// match() from more than one thread.
class OrbRansacMatching : public IMatchingStage {
public:
    explicit OrbRansacMatching(int num_features = 500);

    void setFeatureMask(const cv::Mat& mask) override { feature_mask_ = mask; }

    MatchResult match(const cv::Mat& frame, const ReferenceCrop& crop) override;

private:
    cv::Ptr<cv::ORB> orb_detector_;
    cv::Mat feature_mask_;  // applied only to the query frame, not reference crops

    struct CachedFeatures {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        bool valid = false;
    };
    mutable std::unordered_map<const uchar*, CachedFeatures> feature_cache_;
    const CachedFeatures* getCachedFeatures(const cv::Mat& image) const;
};
