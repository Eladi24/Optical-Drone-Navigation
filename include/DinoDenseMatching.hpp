#pragma once
#include "RetrievalStage.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>
#include <vector>

// STRATEGY.md Phase 2+ "hierarchical semantic/structural matching" direction
// (arXiv:2506.09748), picked up after local-feature retrieval/matching plateaued
// in the fine-tuning pass. An IMatchingStage that matches DENSE DINOv2 patch
// features (the fine-tuned backbone's [1, 256, 384] token grid -- see
// scripts/export_retrieval_backbone.py --dense) between the query frame and one
// reference crop, instead of ORB/XFeat/DISK sparse local features:
//
//   frame tokens  [256, 384]  }  cosine correlation [256, 256]
//   crop  tokens  [256, 384]  }  -> mutual-NN + absolute cosine threshold
//                                -> RANSAC homography on the surviving
//                                   16x16-grid-cell-centre correspondences
//                                -> (H, inlier count)
//
// Same per-candidate IMatchingStage contract as XFeatMatching / DISKLightGlue:
// self-sufficient (does its own extraction, no reliance on a retrieval stage's
// internal state), returns a frame-px -> crop-px homography in the ORIGINAL
// images' pixel space so SplitPipelineEstimator's existing perspectiveTransform
// -> pixelToLatLng refinement works unchanged.
//
// GATE RESULT (flights 01, 10 -- see CLAUDE.md's "Option A" Investigation Log):
// a NEGATIVE result. Given the correct crop, the projected frame-centre is
// genuinely accurate (~33-48 m, vs split_finetuned_disk's 617 m raw) -- the
// fine-tuned dense representation carries the geometry. But crop *discrimination*
// is the ceiling and cannot be raised by anything tried this session: plain
// mutual-NN separates the true crop from its hard retrieval-neighbours only
// ~28-33% of the time, and neither a finer 448/32x32 grid (+5 pts) nor a
// hand-rolled GMS-style spatial-consistency filter (flat) moved it. Flight 01
// gate: recall@1 2.6% / invalid 72.7% / PDM@5 718 m, well behind
// split_finetuned_disk. Kept wired (CLI split_finetuned_dino) as a documented
// negative result -- the discrimination gap is exactly what arXiv:2506.09748's
// LEARNED 4D neighbourhood-consensus module exists to close, the remaining
// unexplored lever.
//
// Caches BOTH the query-frame and the per-crop extraction (keyed by image.data),
// same rationale as XFeatMatching: SplitPipelineEstimator calls match() with the
// same drone_view object once per retrieved candidate.
class DinoDenseMatching : public IMatchingStage {
public:
    explicit DinoDenseMatching(
        const std::string& model_path = "models/dinov2_s_finetuned_dense.onnx",
        float cos_threshold = 0.50f);

    void setFeatureMask(const cv::Mat& mask) override { feature_mask_ = mask; }
    MatchResult match(const cv::Mat& frame, const ReferenceCrop& crop) override;

private:
    struct DenseFeatures {
        std::vector<cv::Point2f> grid_pts;  // patch-grid centres in the 224x224 resize space
                                            // (match() rescales the final homography to orig px)
        cv::Mat descriptors;                // CV_32F, N x 384, L2-normalized per row
        bool valid = false;
    };

    DenseFeatures extract(const cv::Mat& image, const cv::Mat& mask = cv::Mat());
    const DenseFeatures* getCachedFeatures(const cv::Mat& image, const cv::Mat& mask = cv::Mat());

    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo mem_info_;
    float cos_threshold_;
    cv::Mat feature_mask_;

    mutable std::unordered_map<const uchar*, DenseFeatures> feature_cache_;
};
