// Learned matchers (XFeat, DISK+LightGlue) cache the query frame's features. Regression for the
// 2026-11-17 stale-cache bug: the cache was keyed by cv::Mat::data, the per-frame buffer comes
// back at the same address every frame, so every frame was matched with frame 0's keypoints.
// IMatchingStage::beginFrame() must evict it. Needs the ONNX models in models/ (gitignored), so
// this suite is labelled local_only and skips itself when they're absent (CI).
#include "XFeatMatching.hpp"
#include "DiskLightGlueMatching.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <functional>
#include <memory>

namespace {
const std::string kModels = std::string(DRONENAV_SOURCE_DIR) + "/models/";

bool haveModels(std::initializer_list<const char*> names)
{
    for (const char* n : names)
        if (!std::filesystem::exists(kModels + n)) return false;
    return true;
}

// Frame = map shifted by (dx, dy) relative to the crop, so the frame->crop homography is a pure
// translation of (+dx, +dy).
void checkFreshFeaturesPerFrame(IMatchingStage& matcher)
{
    cv::Mat map = testutil::makeTexture(900, 900, 99);
    ReferenceCrop crop({0.0, 0.0}, map(cv::Rect(300, 300, 300, 300)).clone());
    cv::Mat buffer(300, 300, CV_8UC3);
    const uchar* addr = buffer.data;
    for (const cv::Point shift : {cv::Point(20, 0), cv::Point(0, 25), cv::Point(-30, -15)}) {
        map(cv::Rect(300 + shift.x, 300 + shift.y, 300, 300)).copyTo(buffer);
        ASSERT_EQ(buffer.data, addr);  // same address, new content: the bug's trigger
        matcher.beginFrame(buffer);
        MatchResult m = matcher.match(buffer, crop);
        ASSERT_FALSE(m.homography.empty()) << "shift " << shift;
        EXPECT_GT(m.inliers, 20);
        EXPECT_NEAR(m.homography.at<double>(0, 2), shift.x, 3.0) << "shift " << shift;
        EXPECT_NEAR(m.homography.at<double>(1, 2), shift.y, 3.0) << "shift " << shift;
    }
}
}  // namespace

TEST(MatcherCache, XFeatUsesEachFramesOwnFeatures)
{
    if (!haveModels({"xfeat.onnx"})) GTEST_SKIP() << "models/xfeat.onnx not present";
    XFeatMatching matcher(kModels + "xfeat.onnx");
    checkFreshFeaturesPerFrame(matcher);
}

TEST(MatcherCache, DiskLightGlueUsesEachFramesOwnFeatures)
{
    if (!haveModels({"disk.onnx", "disk_lightglue.onnx"})) GTEST_SKIP() << "DISK models not present";
    DiskLightGlueMatching matcher(kModels + "disk.onnx", kModels + "disk_lightglue.onnx");
    checkFreshFeaturesPerFrame(matcher);
}
