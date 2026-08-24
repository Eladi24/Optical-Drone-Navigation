#pragma once
#include "PositionEstimation.hpp"
#include "RetrievalStage.hpp"
#include "ThreadPool.hpp"
#include <memory>
#include <string>

// STRATEGY.md Phase 2: composes an IRetrievalStage + IMatchingStage behind
// the existing IPositionEstimator interface, so VideoProcessing.cpp drives
// it exactly like ORBFeatureEstimator/SIFTFeatureEstimator/etc -- see
// CLAUDE.md's Phase 2 Investigation Log for the full design rationale.
// Per-frame flow: retrieve top-k candidate crop indices, run the matching
// stage on each, keep the best by inlier count, then apply the same
// homography-based continuous-position refinement and
// confidence-from-inliers mapping ORBFeatureEstimator already uses
// (STRATEGY.md Sec 4.2) -- duplicated here rather than shared since it's a
// small, self-contained block and the two estimators have no natural common
// base beyond IPositionEstimator.
//
// The per-candidate matching loop runs on the ThreadPool below (one candidate per task, mirroring
// ORBFeatureEstimator's own ThreadPool+SyncBarrier idiom) ONLY when matching_->isThreadSafe()
// returns true; otherwise it falls back to the original sequential loop, byte-for-byte identical
// to every existing split/split_deit/split_xfeat/split_learned result already recorded in
// CLAUDE.md. Added specifically because CPU-only DISK+LightGlue matching (split_disk) is slow
// enough per candidate that a full multi-flight gate needs the real per-frame parallelism across
// candidates -- see DiskLightGlueMatching.hpp/cpp and CLAUDE.md's DISK+LightGlue Investigation Log.
class SplitPipelineEstimator : public IPositionEstimator {
public:
    SplitPipelineEstimator(std::unique_ptr<IRetrievalStage> retrieval,
                            std::unique_ptr<IMatchingStage> matching,
                            std::string name,
                            int top_k = 10,
                            int num_threads = 4);

    void precompute(const std::vector<ReferenceCrop>& crops) override;
    void setFeatureMask(const cv::Mat& mask) override;
    void setGeoReference(double mpp, double meters_per_degree_lat,
                          double meters_per_degree_lng) override;

    PositionEstimate estimatePosition(
        const cv::Mat& drone_view,
        const std::vector<ReferenceCrop>& reference_crops,
        const std::pair<double, double>& last_position = {0.0, 0.0}
    ) override;

    std::string getName() const override { return name_; }

private:
    std::unique_ptr<IRetrievalStage> retrieval_;
    std::unique_ptr<IMatchingStage> matching_;
    std::string name_;
    int top_k_;

    // Only ever used when matching_->isThreadSafe() -- unused (but harmlessly constructed) for
    // every other matching stage, matching ORBFeatureEstimator's own always-constructed pool_.
    ThreadPool pool_;

    double mpp_ = 0.0;
    double meters_per_degree_lat_ = 0.0;
    double meters_per_degree_lng_ = 0.0;
};
