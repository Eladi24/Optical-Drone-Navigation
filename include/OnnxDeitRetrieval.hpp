#pragma once
#include "RetrievalStage.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// STRATEGY.md Phase 2 learned retrieval: ranks reference crops by cosine
// similarity between L2-normalized embeddings from a frozen, ImageNet-
// pretrained DeiT-Tiny-Distilled backbone (ONNX, CPU inference via ONNX
// Runtime -- see CLAUDE.md's Investigation Log for the model choice
// rationale, the ONNX Runtime C++ build integration, and
// scripts/export_retrieval_backbone.py for how the model file is produced).
//
// Brute-force linear scan over crop embeddings, not an ANN index -- deliberate:
// reference-crop databases in this project are tens to a few hundred crops per
// flight, an exact scan is already fast and avoids adding a second heavy C++
// dependency (faiss) to a repo with no package-manager scaffolding.
class OnnxDeitRetrieval : public IRetrievalStage {
public:
    explicit OnnxDeitRetrieval(const std::string& model_path = "models/deit_tiny_retrieval.onnx");

    void buildIndex(const std::vector<ReferenceCrop>& crops) override;
    void setFeatureMask(const cv::Mat& mask) override { feature_mask_ = mask; }
    std::vector<int> retrieve(const cv::Mat& frame, int k) override;

private:
    // Resize to the model's fixed 224x224 input, ImageNet-normalize, run the
    // ONNX session, L2-normalize the output embedding (so retrieval reduces
    // to a plain dot product). mask (query frame only, never a reference
    // crop -- same convention every other estimator in this codebase
    // follows) is applied in pixel space (zero-filled) before resize, since
    // a ViT forward pass has no native "ignore these pixels" input the way
    // detectAndCompute()'s keypoint mask does.
    std::vector<float> embed(const cv::Mat& image, const cv::Mat& mask = cv::Mat());

    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo mem_info_;
    cv::Mat feature_mask_;
    std::vector<std::vector<float>> crop_embeddings_;  // L2-normalized
    std::vector<int> valid_crop_indices_;               // indices into the original crops vector (skips empty images)
};
