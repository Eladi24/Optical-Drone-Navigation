#pragma once
#include "RetrievalStage.hpp"
#include <onnxruntime_cxx_api.h>
#include <pthread.h>
#include <string>
#include <unordered_map>
#include <vector>

// STRATEGY.md Phase 2 learned matching, second candidate (XFeat was the first -- see
// XFeatMatching.hpp): DISK (verlab-independent, kornia.feature.DISK weights) local-feature
// extractor + LightGlue (cvg/LightGlue, attention-based) matcher, both Apache-2.0, exported to
// ONNX via a pinned tag of fabio-sim/LightGlue-ONNX (v0.1.0 -- see
// scripts/export_disk_lightglue.py's module docstring for why that tag specifically, and why
// its lightglue_onnx/ code is vendored rather than pip-installed).
//
// ARCHITECTURAL DEVIATION FROM XFeatMatching, DELIBERATE: this is TWO ONNX graphs, not one.
// DISK's own NMS-based keypoint selection produces a genuinely VARIABLE number of keypoints per
// image (up to a --max-keypoints cap) -- confirmed empirically via the export script's
// --validate: real/busy content routinely saturates at the cap, while low-texture content sits
// well below it. Every per-image keypoint count here is read from the ACTUAL ONNX Runtime
// output shape at call time (Ort::Value::GetTensorTypeAndShapeInfo()), never assumed fixed like
// XFeatMatching's kTopK. Matching itself is NOT done in C++ at all -- LightGlue is its own
// attention-based matcher network (the second ONNX graph), taking both images' keypoints+
// descriptors and returning match indices directly (already filtered by the model's own
// filter_threshold=0.1, baked into the graph -- no cv::BFMatcher/ratio-test/cosine-threshold
// logic needed here, unlike XFeatMatching's mutual-NN+cosine-threshold C++ logic).
//
// Same per-candidate caching contract as XFeatMatching (SplitPipelineEstimator calls match()
// with the SAME drone_view object once per retrieved candidate -- see
// SplitPipelineEstimator.cpp), same mask-handling contract (pixel-zero pre-resize + hard
// keypoint-position post-filter, query frame only), same never-evicted per-image cache keyed by
// cv::Mat::data.
class DiskLightGlueMatching : public IMatchingStage {
public:
    explicit DiskLightGlueMatching(const std::string& extractor_path = "models/disk.onnx",
                                    const std::string& lightglue_path = "models/disk_lightglue.onnx");
    ~DiskLightGlueMatching() override;
    DiskLightGlueMatching(const DiskLightGlueMatching&) = delete;
    DiskLightGlueMatching& operator=(const DiskLightGlueMatching&) = delete;

    void setFeatureMask(const cv::Mat& mask) override { feature_mask_ = mask; }
    MatchResult match(const cv::Mat& frame, const ReferenceCrop& crop) override;

    // match() is safe to call concurrently on this instance -- feature_cache_ is protected by
    // cache_rwlock_ (same double-checked-locking pattern as ORBFeatureEstimator's own
    // getCachedFeatures()), and Ort::Session::Run() is documented thread-safe for concurrent
    // calls on the same session. Needed in practice, not just possible in principle: CPU-only
    // LightGlue inference makes a full multi-flight gate impractical without exploiting the
    // real per-frame candidate-level parallelism SplitPipelineEstimator can now offer -- see
    // CLAUDE.md's DISK+LightGlue Investigation Log.
    bool isThreadSafe() const override { return true; }

private:
    struct ExtractedFeatures {
        std::vector<cv::Point2f> keypoints_orig;  // rescaled back to the ORIGINAL image's pixel space
        std::vector<cv::Point2f> keypoints_norm;  // normalized in the INPUT_SIZE x INPUT_SIZE frame,
                                                   // ready to feed directly to the LightGlue matcher
                                                   // (formula: (kpt - size/2) / (max(w,h)/2), matching
                                                   // scripts/export_disk_lightglue.py's
                                                   // normalize_keypoints_np verbatim)
        cv::Mat descriptors;                      // CV_32F, N x 128, already L2-normalized by DISK itself
        bool valid = false;
    };

    // Same mask contract as XFeatMatching::extract() -- pixel-zeroed pre-resize (discourages
    // detections there) plus a hard post-filter on rescaled keypoint positions (matches ORB's
    // exact "mask excludes this region" contract, not just a soft discouragement). Does its own
    // ONNX Runtime calls but touches no shared state -- safe to run unlocked, concurrently, on
    // different images from different threads; only feature_cache_ itself needs synchronizing
    // (done in getCachedFeatures(), not here), same division of responsibility as
    // ORBFeatureEstimator's own extract-outside-lock/insert-inside-lock split.
    ExtractedFeatures extract(const cv::Mat& image, const cv::Mat& mask = cv::Mat());

    // Thread-safe: double-checked rwlock locking, mirrors ORBFeatureEstimator::getCachedFeatures()
    // exactly (shared rdlock fast path for cache hits, exclusive wrlock only on a genuine miss,
    // with a second lookup after re-acquiring the lock in case another thread inserted the same
    // key while this one was computing). Pointers to unordered_map values remain valid across
    // insertion/rehash by the container's own standard guarantee (only erasure invalidates them,
    // and nothing here ever erases) -- the lock exists to serialize concurrent structural
    // mutation, not for pointer-stability reasons.
    const ExtractedFeatures* getCachedFeatures(const cv::Mat& image, const cv::Mat& mask = cv::Mat());

    Ort::Env env_;
    Ort::Session extractor_session_;
    Ort::Session lightglue_session_;
    Ort::MemoryInfo mem_info_;
    cv::Mat feature_mask_;

    mutable pthread_rwlock_t cache_rwlock_;
    mutable std::unordered_map<const uchar*, ExtractedFeatures> feature_cache_;
};
