#pragma once
#include <string>
#include <vector>

// Pure VLAD math used by OnnxVladRetrieval, split out (2026-11-23) so it can be unit-tested
// without an ONNX session or model files -- in particular for C++ <-> Python parity against
// scripts/proto_vlad_retrieval.py's vlad_encode (tests/test_vlad.cpp). No ONNX dependency.
namespace vlad {

// In-place L2 normalization; a (near-)zero vector is left unchanged.
void l2normalize(float* v, int n);

struct Vocabulary {
    int k = 0;                 // number of centroids
    int dim = 0;               // descriptor dimension
    std::vector<float> data;   // [k * dim], row-major, each row L2-normalized
};

// Reads the binary written by scripts/export_vlad_vocab.py: int32 magic 'VLAD'
// (0x44414C56, little-endian), int32 k, int32 D, then k*D float32 row-major.
// Rows are re-normalized defensively. Throws std::runtime_error on a missing file,
// bad magic, non-positive sizes or truncation.
Vocabulary readVocabulary(const std::string& path);

// VLAD-encode L2-normalized patch descriptors against L2-normalized centroids:
// hard-assign each patch to its nearest centroid (max dot product == min L2 for unit
// vectors), sum residuals per centroid, intra-normalize each block, signed-sqrt power
// normalization, global L2. Output length k*dim. Mirrors proto_vlad_retrieval.vlad_encode.
std::vector<float> encode(const std::vector<std::vector<float>>& patch_descs,
                          const std::vector<float>& vocab, int k, int dim);

}  // namespace vlad
