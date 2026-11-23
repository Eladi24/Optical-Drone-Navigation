#include "VladMath.hpp"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace vlad {

namespace {
constexpr int32_t kVocabMagic = 0x44414C56;  // 'VLAD' little-endian
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

Vocabulary readVocabulary(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("vlad::readVocabulary: cannot open vocab file " + path +
                                     " (produce it with scripts/export_vlad_vocab.py)");
    int32_t header[3];
    f.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!f || header[0] != kVocabMagic)
        throw std::runtime_error("vlad::readVocabulary: bad vocab header in " + path);
    Vocabulary voc;
    voc.k = header[1];
    voc.dim = header[2];
    if (voc.k <= 0 || voc.dim <= 0)
        throw std::runtime_error("vlad::readVocabulary: non-positive k/dim in " + path);
    voc.data.resize(static_cast<size_t>(voc.k) * voc.dim);
    f.read(reinterpret_cast<char*>(voc.data.data()),
           static_cast<std::streamsize>(voc.data.size() * sizeof(float)));
    if (!f) throw std::runtime_error("vlad::readVocabulary: vocab file truncated: " + path);
    // Centroids are written already L2-normalized; re-normalize defensively so the
    // nearest-centroid-by-dot-product step in encode() is exactly nearest-by-L2.
    for (int c = 0; c < voc.k; ++c) l2normalize(voc.data.data() + static_cast<size_t>(c) * voc.dim, voc.dim);
    return voc;
}

std::vector<float> encode(const std::vector<std::vector<float>>& patch_descs,
                          const std::vector<float>& vocab, int k, int dim)
{
    // Accumulate residuals per centroid (mirrors scripts/proto_vlad_retrieval.py's vlad_encode).
    std::vector<float> V(static_cast<size_t>(k) * dim, 0.0f);
    for (const auto& p : patch_descs) {
        int best_c = 0;
        double best_dot = -2.0;
        for (int c = 0; c < k; ++c) {
            const float* cen = vocab.data() + static_cast<size_t>(c) * dim;
            double dot = 0.0;
            for (int d = 0; d < dim; ++d) dot += static_cast<double>(p[d]) * cen[d];
            if (dot > best_dot) { best_dot = dot; best_c = c; }
        }
        float* acc = V.data() + static_cast<size_t>(best_c) * dim;
        const float* cen = vocab.data() + static_cast<size_t>(best_c) * dim;
        for (int d = 0; d < dim; ++d) acc[d] += p[d] - cen[d];
    }
    // Intra-normalization (L2 per centroid block).
    for (int c = 0; c < k; ++c) l2normalize(V.data() + static_cast<size_t>(c) * dim, dim);
    // Signed square-root power-normalization, then global L2.
    for (float& x : V) x = (x >= 0.0f ? 1.0f : -1.0f) * std::sqrt(std::fabs(x));
    l2normalize(V.data(), static_cast<int>(V.size()));
    return V;
}

}  // namespace vlad
