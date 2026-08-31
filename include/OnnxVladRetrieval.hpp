#pragma once
#include "RetrievalStage.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// RESULTS.md route #2 / AnyLoc (arXiv:2308.00688): retrieval by VLAD-aggregating
// the DENSE per-patch features of a DINOv2 backbone, instead of a single pooled
// embedding (OnnxDeitRetrieval). Measured in Python first
// (scripts/proto_vlad_retrieval.py) -- on UAV-VisLoc held-out flights, VLAD over
// the fine-tuned dense features (k=64) gives the best pure-embedding recall@k the
// project has produced (13.5 / 33.0 / 45.2 vs. fine-tuned pooled's 13.0 / 30.8 /
// 41.5), and frozen-backbone VLAD still clearly beats frozen pooled (+28%
// recall@1) with no training. See CLAUDE.md's "VLAD retrieval" Investigation Log.
//
// Pipeline per image:
//   1. dense patch tokens from the ONNX model ('patch_tokens', [1, 256, 384]),
//      L2-normalized per patch
//   2. hard-assign each patch to the nearest of k vocabulary centroids
//   3. sum residuals (patch - centroid) per centroid, intra-normalize each block
//   4. concatenate -> [k*384], signed-sqrt power-normalize, global L2
// Retrieval is then a plain dot product (cosine) between VLAD vectors, same
// brute-force linear scan as OnnxDeitRetrieval and for the same reason (tens to a
// few hundred crops per flight -- not ANN-index territory).
//
// The vocabulary is k-means over dev-flight (01/03) reference-crop patch
// descriptors, produced by scripts/export_vlad_vocab.py as a small binary
// (models/vlad_vocab_*.bin: int32 magic 'VLAD', int32 k, int32 D, then k*D
// float32 row-major, centroids already L2-normalized).
class OnnxVladRetrieval : public IRetrievalStage {
public:
    explicit OnnxVladRetrieval(
        const std::string& model_path = "models/dinov2_s_finetuned_dense.onnx",
        const std::string& vocab_path = "models/vlad_vocab_finetuned_k64.bin");

    void buildIndex(const std::vector<ReferenceCrop>& crops) override;
    void setFeatureMask(const cv::Mat& mask) override { feature_mask_ = mask; }
    std::vector<int> retrieve(const cv::Mat& frame, int k) override;

private:
    // dense patch tokens [num_tokens][desc_dim], L2-normalized per row. mask
    // (query frame only -- same convention as every other estimator here) is
    // zero-filled in pixel space before the resize, matching OnnxDeitRetrieval.
    std::vector<std::vector<float>> embedDense(const cv::Mat& image, const cv::Mat& mask = cv::Mat());

    // VLAD-aggregate patch descriptors against vocab_ -> [k*desc_dim], power- and
    // L2-normalized (so retrieval is a dot product).
    std::vector<float> vladEncode(const std::vector<std::vector<float>>& patch_descs) const;

    void loadVocabulary(const std::string& vocab_path);

    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo mem_info_;

    int k_ = 0;                                  // vocabulary size
    int desc_dim_ = 0;                            // per-patch descriptor dimension (384)
    std::vector<float> vocab_;                    // [k_ * desc_dim_], row-major, L2-normalized

    cv::Mat feature_mask_;
    std::vector<std::vector<float>> crop_vlad_;   // [k_*desc_dim_] per valid crop, power+L2-normalized
    std::vector<int> valid_crop_indices_;
};
