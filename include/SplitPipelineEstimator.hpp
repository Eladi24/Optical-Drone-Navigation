#pragma once
#include "PositionEstimation.hpp"
#include "RetrievalStage.hpp"
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
class SplitPipelineEstimator : public IPositionEstimator {
public:
    SplitPipelineEstimator(std::unique_ptr<IRetrievalStage> retrieval,
                            std::unique_ptr<IMatchingStage> matching,
                            std::string name,
                            int top_k = 10);

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

    double mpp_ = 0.0;
    double meters_per_degree_lat_ = 0.0;
    double meters_per_degree_lng_ = 0.0;
};
