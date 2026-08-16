#include "OnnxDeitRetrieval.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>

namespace {
constexpr int kInputSize = 224;  // must match scripts/export_retrieval_backbone.py's fixed export shape
constexpr float kImageNetMean[3] = {0.485f, 0.456f, 0.406f};  // RGB order
constexpr float kImageNetStd[3]  = {0.229f, 0.224f, 0.225f};

Ort::SessionOptions makeSessionOptions()
{
    Ort::SessionOptions opts;
    // This codebase's own ThreadPool already parallelizes across frames/
    // candidates elsewhere (ORBFeatureEstimator, etc.); avoid ONNX Runtime's
    // default intra-op threading oversubscribing on top of that.
    opts.SetIntraOpNumThreads(1);
    return opts;
}
}  // namespace

OnnxDeitRetrieval::OnnxDeitRetrieval(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "OnnxDeitRetrieval")
    , session_(env_, model_path.c_str(), makeSessionOptions())
    , mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
}

std::vector<float> OnnxDeitRetrieval::embed(const cv::Mat& image, const cv::Mat& mask)
{
    cv::Mat working = image;
    if (!mask.empty() && mask.size() == image.size()) {
        working = image.clone();
        working.setTo(cv::Scalar(0, 0, 0), mask == 0);
    }

    cv::Mat resized;
    cv::resize(working, resized, cv::Size(kInputSize, kInputSize));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<float> input(static_cast<size_t>(3) * kInputSize * kInputSize);
    const int plane = kInputSize * kInputSize;
    for (int y = 0; y < kInputSize; ++y) {
        const cv::Vec3f* row = rgb.ptr<cv::Vec3f>(y);
        for (int x = 0; x < kInputSize; ++x) {
            for (int c = 0; c < 3; ++c) {
                input[static_cast<size_t>(c) * plane + y * kInputSize + x] =
                    (row[x][c] - kImageNetMean[c]) / kImageNetStd[c];
            }
        }
    }

    std::array<int64_t, 4> shape{1, 3, kInputSize, kInputSize};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_, input.data(), input.size(), shape.data(), shape.size());

    const char* input_names[]  = {"image"};
    const char* output_names[] = {"embedding"};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                 output_names, 1);

    const float* out_data = outputs[0].GetTensorData<float>();
    size_t embed_dim = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> embedding(out_data, out_data + embed_dim);

    double norm = 0.0;
    for (float v : embedding) norm += static_cast<double>(v) * v;
    norm = std::sqrt(norm);
    if (norm > 1e-12) {
        for (float& v : embedding) v = static_cast<float>(v / norm);
    }
    return embedding;
}

void OnnxDeitRetrieval::buildIndex(const std::vector<ReferenceCrop>& crops)
{
    crop_embeddings_.clear();
    valid_crop_indices_.clear();
    crop_embeddings_.reserve(crops.size());
    valid_crop_indices_.reserve(crops.size());

    for (int i = 0; i < static_cast<int>(crops.size()); ++i) {
        if (crops[i].image.empty()) continue;
        crop_embeddings_.push_back(embed(crops[i].image));
        valid_crop_indices_.push_back(i);
    }
}

std::vector<int> OnnxDeitRetrieval::retrieve(const cv::Mat& frame, int k)
{
    if (frame.empty() || crop_embeddings_.empty()) return {};

    std::vector<float> frame_embedding = embed(frame, feature_mask_);

    // (cosine similarity, original crop index) -- both sides pre-normalized,
    // so a dot product is the cosine similarity directly.
    std::vector<std::pair<double, int>> scored;
    scored.reserve(crop_embeddings_.size());
    for (size_t i = 0; i < crop_embeddings_.size(); ++i) {
        const auto& emb = crop_embeddings_[i];
        double score = 0.0;
        for (size_t d = 0; d < emb.size(); ++d) score += static_cast<double>(frame_embedding[d]) * emb[d];
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
