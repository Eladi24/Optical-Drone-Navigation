#pragma once
#include "RetrievalStage.hpp"

// Classical global-descriptor retrieval baseline -- STRATEGY.md Phase 2's
// "implement retrieval with a classical global descriptor first, purely to
// validate the plumbing" step. Not intended to be a strong retrieval method
// on its own (that's the learned backbone's job in a later pass); HSV color
// histogram per crop, ranked per frame by histogram similarity, is a
// standard, well-understood CBIR baseline -- simplest correct thing that
// isn't trivially broken.
class GlobalHistogramRetrieval : public IRetrievalStage {
public:
    void buildIndex(const std::vector<ReferenceCrop>& crops) override;
    void setFeatureMask(const cv::Mat& mask) override { feature_mask_ = mask; }
    std::vector<int> retrieve(const cv::Mat& frame, int k) override;

private:
    // mask is applied only when hashing the query frame, never a reference
    // crop -- same "mask excludes HUD/logo regions from the drone view
    // only" convention every other estimator in this codebase follows.
    static cv::Mat computeHistogram(const cv::Mat& image, const cv::Mat& mask = cv::Mat());

    cv::Mat feature_mask_;
    std::vector<cv::Mat> crop_histograms_;
    std::vector<int> valid_crop_indices_;  // indices into the original crops vector (skips empty images)
};
