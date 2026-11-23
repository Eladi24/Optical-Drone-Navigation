// VLAD math: C++ <-> Python parity. OnnxVladRetrieval's encoder must mirror
// scripts/proto_vlad_retrieval.py's vlad_encode exactly -- every retrieval number measured in
// Python is only valid for the C++ pipeline if they agree. Fixture: tests/fixtures/make_vlad_fixture.py.
#include "VladMath.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>

namespace {
const std::string kFixtureDir = DRONENAV_FIXTURE_DIR;

struct Case {
    int P = 0, D = 0, k = 0;
    std::vector<std::vector<float>> patches;
    std::vector<float> expected;
};

Case readCase()
{
    Case c;
    std::ifstream f(kFixtureDir + "/vlad_small_case.bin", std::ios::binary);
    if (!f) throw std::runtime_error("missing fixture vlad_small_case.bin");
    int32_t hdr[3];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    c.P = hdr[0]; c.D = hdr[1]; c.k = hdr[2];
    c.patches.assign(c.P, std::vector<float>(c.D));
    for (auto& p : c.patches) f.read(reinterpret_cast<char*>(p.data()), c.D * sizeof(float));
    c.expected.resize(static_cast<size_t>(c.k) * c.D);
    f.read(reinterpret_cast<char*>(c.expected.data()), c.expected.size() * sizeof(float));
    if (!f) throw std::runtime_error("truncated fixture vlad_small_case.bin");
    return c;
}

void writeBytes(const std::string& path, const std::vector<int32_t>& header, size_t n_floats)
{
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(header.data()), header.size() * sizeof(int32_t));
    std::vector<float> zeros(n_floats, 1.0f);
    f.write(reinterpret_cast<const char*>(zeros.data()), zeros.size() * sizeof(float));
}
}  // namespace

TEST(Vlad, ReadsTheExportedVocabularyFormat)
{
    vlad::Vocabulary voc = vlad::readVocabulary(kFixtureDir + "/vlad_small_vocab.bin");
    EXPECT_EQ(voc.k, 8);
    EXPECT_EQ(voc.dim, 16);
    ASSERT_EQ(voc.data.size(), 8u * 16u);
    for (int c = 0; c < voc.k; ++c) {
        double n = 0;
        for (int d = 0; d < voc.dim; ++d) n += double(voc.data[c * voc.dim + d]) * voc.data[c * voc.dim + d];
        EXPECT_NEAR(std::sqrt(n), 1.0, 1e-6);
    }
}

TEST(Vlad, EncodeMatchesPythonReference)
{
    Case c = readCase();
    vlad::Vocabulary voc = vlad::readVocabulary(kFixtureDir + "/vlad_small_vocab.bin");
    ASSERT_EQ(voc.k, c.k);
    ASSERT_EQ(voc.dim, c.D);
    std::vector<float> v = vlad::encode(c.patches, voc.data, voc.k, voc.dim);
    ASSERT_EQ(v.size(), c.expected.size());
    for (size_t i = 0; i < v.size(); ++i) EXPECT_NEAR(v[i], c.expected[i], 1e-5) << "element " << i;
}

TEST(Vlad, EncodeIsUnitNormAndEmptyCentroidsStayZero)
{
    Case c = readCase();
    vlad::Vocabulary voc = vlad::readVocabulary(kFixtureDir + "/vlad_small_vocab.bin");
    std::vector<float> v = vlad::encode(c.patches, voc.data, voc.k, voc.dim);
    const double norm = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
    EXPECT_NEAR(norm, 1.0, 1e-6);
    // The fixture leaves centroids 6 and 7 without patches (see make_vlad_fixture.py).
    for (int c_idx : {6, 7})
        for (int d = 0; d < voc.dim; ++d) EXPECT_EQ(v[c_idx * voc.dim + d], 0.0f);
}

TEST(Vlad, RejectsBadVocabularyFiles)
{
    testutil::ScopedTempCwd tmp;
    EXPECT_THROW(vlad::readVocabulary("missing.bin"), std::runtime_error);
    writeBytes("bad_magic.bin", {0x12345678, 8, 16}, 8 * 16);
    EXPECT_THROW(vlad::readVocabulary("bad_magic.bin"), std::runtime_error);
    writeBytes("truncated.bin", {0x44414C56, 8, 16}, 8 * 16 - 1);
    EXPECT_THROW(vlad::readVocabulary("truncated.bin"), std::runtime_error);
    writeBytes("zero_k.bin", {0x44414C56, 0, 16}, 0);
    EXPECT_THROW(vlad::readVocabulary("zero_k.bin"), std::runtime_error);
}
