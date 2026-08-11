#include "HistogramRetrieval.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace {
constexpr int kHBins = 30;
constexpr int kSBins = 32;
}

cv::Mat GlobalHistogramRetrieval::computeHistogram(const cv::Mat& image, const cv::Mat& mask)
{
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    int hist_size[]     = {kHBins, kSBins};
    float h_range[]      = {0, 180};
    float s_range[]      = {0, 256};
    const float* ranges[] = {h_range, s_range};
    int channels[]       = {0, 1};

    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, mask, hist, 2, hist_size, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    return hist;
}

void GlobalHistogramRetrieval::buildIndex(const std::vector<ReferenceCrop>& crops)
{
    crop_histograms_.clear();
    valid_crop_indices_.clear();
    crop_histograms_.reserve(crops.size());
    valid_crop_indices_.reserve(crops.size());

    for (int i = 0; i < static_cast<int>(crops.size()); ++i) {
        if (crops[i].image.empty()) continue;
        crop_histograms_.push_back(computeHistogram(crops[i].image));
        valid_crop_indices_.push_back(i);
    }
}

std::vector<int> GlobalHistogramRetrieval::retrieve(const cv::Mat& frame, int k)
{
    if (frame.empty() || crop_histograms_.empty()) return {};

    cv::Mat frame_hist = computeHistogram(frame, feature_mask_);

    // (similarity, original crop index) -- HISTCMP_CORREL: higher is more similar.
    std::vector<std::pair<double, int>> scored;
    scored.reserve(crop_histograms_.size());
    for (size_t i = 0; i < crop_histograms_.size(); ++i) {
        double score = cv::compareHist(frame_hist, crop_histograms_[i], cv::HISTCMP_CORREL);
        scored.push_back({score, valid_crop_indices_[i]});
    }

    int n = std::min<int>(k, static_cast<int>(scored.size()));
    std::partial_sort(scored.begin(), scored.begin() + n, scored.end(),
                       [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<int> result;
    result.reserve(n);
    for (int i = 0; i < n; ++i) result.push_back(scored[i].second);
    return result;
}
