// End-to-end retrieval -> matching -> homography -> position on a procedural map, with the
// classical stages only (no ONNX, no models): GlobalHistogramRetrieval + OrbRansacMatching
// behind SplitPipelineEstimator. Regressions for:
//  - the raw position must come from the homography, not the crop centroid;
//  - every "no measurement" exit must report measurement_valid = false (a fused RANSAC-failure
//    fallback is hollow self-confirmation of the Kalman prediction);
//  - per-frame matcher caches must not reuse the previous frame's features when the frame
//    buffer is re-allocated at the same address (the 2026-11-17 stale-cache bug).
#include "SplitPipelineEstimator.hpp"
#include "HistogramRetrieval.hpp"
#include "OrbRansacMatching.hpp"
#include "VideoProcessing.hpp"
#include "CoordinateUtils.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>

namespace {
constexpr double kLat = 31.0, kLng = 120.0;
constexpr double kMpp = 1.0;
const double kMpdLat = testutil::kMetersPerDegLat;
const double kMpdLng = testutil::metersPerDegLng(kLat);

struct World {
    cv::Mat map;
    std::vector<ReferenceCrop> crops;
    int cx, cy;
};

const World& world()
{
    static World w = [] {
        World w;
        w.map = testutil::makeTexture(1000, 1000);
        w.cx = w.map.cols / 2;
        w.cy = w.map.rows / 2;
        w.crops = generateReferenceCropsGrid(w.map, kLat, kLng, w.cx, w.cy, kMpp, kMpdLat, kMpdLng,
                                             /*spacing*/ 100, /*crop*/ kReferenceCropMeters);
        return w;
    }();
    return w;
}

// A "drone view": the map around `centre_px`, rotated by `angle_deg`, kReferenceCropMeters wide.
cv::Mat droneView(const cv::Point2f& centre_px, double angle_deg)
{
    const int take = static_cast<int>(kReferenceCropMeters / kMpp);
    cv::Mat M = cv::getRotationMatrix2D(centre_px, angle_deg, 1.0);
    M.at<double>(0, 2) += take / 2.0 - centre_px.x;
    M.at<double>(1, 2) += take / 2.0 - centre_px.y;
    cv::Mat out;
    cv::warpAffine(world().map, out, M, cv::Size(take, take), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return out;
}

std::unique_ptr<SplitPipelineEstimator> makeEstimator()
{
    auto est = std::make_unique<SplitPipelineEstimator>(
        std::make_unique<GlobalHistogramRetrieval>(), std::make_unique<OrbRansacMatching>(1000),
        "test_split", /*top_k: all crops*/ static_cast<int>(world().crops.size()), /*threads*/ 1);
    est->precompute(world().crops);
    est->setGeoReference(kMpp, kMpdLat, kMpdLng);
    return est;
}

double errorMeters(const PositionEstimate& e, const cv::Point2f& truth_px)
{
    auto truth = CoordinateUtils::pixelToLatLng(static_cast<int>(truth_px.x), static_cast<int>(truth_px.y), kLat,
                                                kLng, world().cx, world().cy, kMpp, kMpdLat, kMpdLng);
    return CoordinateUtils::calculateDistance(truth, e.position, kMpdLat, kMpdLng);
}
}  // namespace

TEST(SplitPipeline, LocalisesAnOffGridViewFromTheHomography)
{
    cv::setNumThreads(0);
    auto est = makeEstimator();
    // 37 m east / 42 m south of the nearest grid point, so the crop centroid alone would be ~56 m off.
    const cv::Point2f truth(437.0f, 542.0f);
    PositionEstimate e = est->estimatePosition(droneView(truth, 12.0), world().crops);
    ASSERT_TRUE(e.measurement_valid);
    EXPECT_GE(e.inliers, 20);
    EXPECT_LT(errorMeters(e, truth), 5.0);
}

TEST(SplitPipeline, FeaturelessFrameIsNotAMeasurement)
{
    cv::setNumThreads(0);
    auto est = makeEstimator();
    cv::Mat blank(300, 300, CV_8UC3, cv::Scalar(90, 90, 90));
    PositionEstimate e = est->estimatePosition(blank, world().crops, {kLat, kLng});
    EXPECT_FALSE(e.measurement_valid);
}

TEST(SplitPipeline, EmptyInputsAreNotAMeasurement)
{
    auto est = makeEstimator();
    EXPECT_FALSE(est->estimatePosition(cv::Mat(), world().crops).measurement_valid);
    EXPECT_FALSE(est->estimatePosition(droneView({500, 500}, 0), {}).measurement_valid);
}

// Same frame buffer (same cv::Mat::data address), different content each frame -- exactly what
// the video loop does. Each frame must be localised from ITS OWN content.
TEST(SplitPipeline, ReusedFrameBufferIsNotAStaleCache)
{
    cv::setNumThreads(0);
    auto est = makeEstimator();
    cv::Mat buffer(300, 300, CV_8UC3);
    const uchar* addr = buffer.data;
    for (const cv::Point2f truth : {cv::Point2f(437, 542), cv::Point2f(612, 388), cv::Point2f(305, 700)}) {
        droneView(truth, -20.0).copyTo(buffer);
        ASSERT_EQ(buffer.data, addr);
        PositionEstimate e = est->estimatePosition(buffer, world().crops);
        ASSERT_TRUE(e.measurement_valid);
        EXPECT_LT(errorMeters(e, truth), 5.0) << "truth px " << truth;
    }
}
