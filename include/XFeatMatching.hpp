#pragma once
#include "RetrievalStage.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>
#include <vector>

// STRATEGY.md Phase 2 learned matching: XFeat (verlab/accelerated_features,
// Apache-2.0) local-feature detector+descriptor, ONNX/CPU inference, wired in
// as a drop-in IMatchingStage replacement for OrbRansacMatching -- same
// per-candidate contract (independently detect+match+RANSAC against one
// crop), same NN+Lowe's-ratio+RANSAC verification logic, ONLY the
// detector/descriptor is learned instead of ORB's classical one. See
// CLAUDE.md's Investigation Log and scripts/export_xfeat.py's docstring for
// why this uses XFeat's dense (fixed top_k) extraction path rather than its
// NMS-based sparse path (the latter has a genuinely data-dependent output
// shape that can't be cleanly ONNX-exported).
//
// Unlike OrbRansacMatching, this ALSO caches the query frame's own
// extraction (keyed by image.data, same map as the per-crop cache) --
// SplitPipelineEstimator calls match() with the SAME drone_view object once
// per retrieved candidate (see SplitPipelineEstimator.cpp), so without this
// a ~1024-keypoint CNN forward pass would otherwise repeat up to top_k times
// per frame. The cache is never evicted (same design choice as
// OrbRansacMatching's own per-crop cache); frame-side entries become dead
// weight after their frame is done, but the resulting growth (roughly
// top_k*64 floats + top_k*2 floats per frame) is small enough over one
// video's frame count not to be worth an eviction policy.
class XFeatMatching : public IMatchingStage {
public:
    explicit XFeatMatching(const std::string& model_path = "models/xfeat.onnx",
                            float score_threshold = 0.05f);

    void setFeatureMask(const cv::Mat& mask) override { feature_mask_ = mask; }
    MatchResult match(const cv::Mat& frame, const ReferenceCrop& crop) override;

private:
    struct ExtractedFeatures {
        std::vector<cv::Point2f> keypoints;  // rescaled back to the ORIGINAL image's pixel space
        cv::Mat descriptors;                 // CV_32F, N x 64, L2-normalized
        bool valid = false;
    };

    // mask (query frame only, never a reference crop) applied two ways: zero-filled
    // in pixel space before the network sees the image (discourages spurious
    // detections there), and as a hard post-filter on returned keypoint positions
    // (matches ORB's exact "mask excludes this region from detection" contract,
    // not just a soft discouragement -- see CLAUDE.md's mask-alignment findings).
    ExtractedFeatures extract(const cv::Mat& image, const cv::Mat& mask = cv::Mat());
    const ExtractedFeatures* getCachedFeatures(const cv::Mat& image, const cv::Mat& mask = cv::Mat());

    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo mem_info_;
    float score_threshold_;
    cv::Mat feature_mask_;

    mutable std::unordered_map<const uchar*, ExtractedFeatures> feature_cache_;
};
