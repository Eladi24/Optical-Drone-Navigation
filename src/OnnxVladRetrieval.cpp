#include "StageProfiler.hpp"
#include "OnnxVladRetrieval.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace {
// Must match scripts/export_retrieval_backbone.py --dense (INPUT_SIZE 224,
// DINOv2-S patch14 -> 16x16 = 256 tokens, 384-dim) and scripts/proto_vlad_retrieval.py.
constexpr int kInputSize = 224;
constexpr int kGrid      = 16;
constexpr int kNumTokens = kGrid * kGrid;  // 256
constexpr int kDescDim   = 384;
constexpr float kImageNetMean[3] = {0.485f, 0.456f, 0.406f};  // RGB order
constexpr float kImageNetStd[3]  = {0.229f, 0.224f, 0.225f};
constexpr int32_t kVocabMagic = 0x44414C56;  // 'VLAD' little-endian

Ort::SessionOptions makeSessionOptions()
{
    Ort::SessionOptions opts;
    // This codebase parallelizes across frames/candidates itself; don't let ONNX
    // Runtime's intra-op pool oversubscribe on top (same as OnnxDeitRetrieval).
    opts.SetIntraOpNumThreads(1);
    return opts;
}

void l2normalize(float* v, int n)
{
    double norm = 0.0;
    for (int i = 0; i < n; ++i) norm += static_cast<double>(v[i]) * v[i];
    norm = std::sqrt(norm);
    if (norm > 1e-12) {
        for (int i = 0; i < n; ++i) v[i] = static_cast<float>(v[i] / norm);
    }
}
}  // namespace

OnnxVladRetrieval::OnnxVladRetrieval(const std::string& model_path, const std::string& vocab_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "OnnxVladRetrieval")
    , session_(env_, model_path.c_str(), makeSessionOptions())
    , mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    loadVocabulary(vocab_path);
}

void OnnxVladRetrieval::loadVocabulary(const std::string& vocab_path)
{
    std::ifstream f(vocab_path, std::ios::binary);
    if (!f) throw std::runtime_error("OnnxVladRetrieval: cannot open vocab file " + vocab_path +
                                     " (produce it with scripts/export_vlad_vocab.py)");
    int32_t header[3];
    f.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!f || header[0] != kVocabMagic)
        throw std::runtime_error("OnnxVladRetrieval: bad vocab header in " + vocab_path);
    k_ = header[1];
    desc_dim_ = header[2];
    if (k_ <= 0 || desc_dim_ != kDescDim)
        throw std::runtime_error("OnnxVladRetrieval: vocab dim mismatch (expected desc_dim " +
                                 std::to_string(kDescDim) + ")");
    vocab_.resize(static_cast<size_t>(k_) * desc_dim_);
    f.read(reinterpret_cast<char*>(vocab_.data()), static_cast<std::streamsize>(vocab_.size() * sizeof(float)));
    if (!f) throw std::runtime_error("OnnxVladRetrieval: vocab file truncated: " + vocab_path);
    // Centroids are written already L2-normalized; re-normalize defensively so the
    // nearest-centroid-by-dot-product step below is exactly nearest-by-L2.
    for (int c = 0; c < k_; ++c) l2normalize(vocab_.data() + static_cast<size_t>(c) * desc_dim_, desc_dim_);
}

std::vector<std::vector<float>> OnnxVladRetrieval::embedDense(const cv::Mat& image, const cv::Mat& mask)
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
        for (int x = 0; x < kInputSize; ++x)
            for (int c = 0; c < 3; ++c)
                input[static_cast<size_t>(c) * plane + y * kInputSize + x] =
                    (row[x][c] - kImageNetMean[c]) / kImageNetStd[c];
    }

    std::array<int64_t, 4> shape{1, 3, kInputSize, kInputSize};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_, input.data(), input.size(), shape.data(), shape.size());

    const char* input_names[]  = {"image"};
    const char* output_names[] = {"patch_tokens"};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                 output_names, 1);

    const float* tok = outputs[0].GetTensorData<float>();  // [1, num_tokens, desc_dim], row-major
    size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    int num_tokens = static_cast<int>(count / kDescDim);
    if (num_tokens <= 0) return {};

    std::vector<std::vector<float>> descs(num_tokens, std::vector<float>(kDescDim));
    for (int t = 0; t < num_tokens; ++t) {
        std::copy(tok + static_cast<size_t>(t) * kDescDim,
                  tok + static_cast<size_t>(t + 1) * kDescDim, descs[t].begin());
        l2normalize(descs[t].data(), kDescDim);
    }
    return descs;
}

std::vector<float> OnnxVladRetrieval::vladEncode(const std::vector<std::vector<float>>& patch_descs) const
{
    // Accumulate residuals per centroid (mirrors scripts/proto_vlad_retrieval.py's vlad_encode).
    std::vector<float> V(static_cast<size_t>(k_) * desc_dim_, 0.0f);
    for (const auto& p : patch_descs) {
        int best_c = 0;
        double best_dot = -2.0;
        for (int c = 0; c < k_; ++c) {
            const float* cen = vocab_.data() + static_cast<size_t>(c) * desc_dim_;
            double dot = 0.0;
            for (int d = 0; d < desc_dim_; ++d) dot += static_cast<double>(p[d]) * cen[d];
            if (dot > best_dot) { best_dot = dot; best_c = c; }
        }
        float* acc = V.data() + static_cast<size_t>(best_c) * desc_dim_;
        const float* cen = vocab_.data() + static_cast<size_t>(best_c) * desc_dim_;
        for (int d = 0; d < desc_dim_; ++d) acc[d] += p[d] - cen[d];
    }
    // Intra-normalization (L2 per centroid block).
    for (int c = 0; c < k_; ++c) l2normalize(V.data() + static_cast<size_t>(c) * desc_dim_, desc_dim_);
    // Signed square-root power-normalization, then global L2.
    for (float& x : V) x = (x >= 0.0f ? 1.0f : -1.0f) * std::sqrt(std::fabs(x));
    l2normalize(V.data(), static_cast<int>(V.size()));
    return V;
}

void OnnxVladRetrieval::buildIndex(const std::vector<ReferenceCrop>& crops)
{
    crop_vlad_.clear();
    valid_crop_indices_.clear();
    crop_vlad_.reserve(crops.size());
    valid_crop_indices_.reserve(crops.size());

    for (int i = 0; i < static_cast<int>(crops.size()); ++i) {
        if (crops[i].image.empty()) continue;
        auto descs = embedDense(crops[i].image);
        if (descs.empty()) continue;
        crop_vlad_.push_back(vladEncode(descs));
        valid_crop_indices_.push_back(i);
    }
}

std::vector<int> OnnxVladRetrieval::retrieve(const cv::Mat& frame, int k)
{
    if (frame.empty() || crop_vlad_.empty()) return {};

    std::vector<std::vector<float>> frame_descs;
    { StageProfiler::ScopedStage s("3a1 retrieval: DINOv2 embed frame"); frame_descs = embedDense(frame, feature_mask_); }
    if (frame_descs.empty()) return {};
    StageProfiler::ScopedStage s_scan("3a2 retrieval: VLAD encode + scan");
    std::vector<float> q = vladEncode(frame_descs);

    std::vector<std::pair<double, int>> scored;
    scored.reserve(crop_vlad_.size());
    for (size_t i = 0; i < crop_vlad_.size(); ++i) {
        const auto& v = crop_vlad_[i];
        double score = 0.0;
        for (size_t d = 0; d < v.size(); ++d) score += static_cast<double>(q[d]) * v[d];
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
