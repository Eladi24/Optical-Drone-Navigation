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

// ---------------------------------------------------------------------------------------------------
// Candidate re-centring (opt-in, 2026-11-24; INVESTIGATION_LOG "av1 dusk matching diagnostic").
// A sparse grid (300 m crops every 200 m) leaves the true crop mis-centred; re-centring must raise
// the winner's inliers, keep the position, and never serve a short-lived crop from a feature cache.
// ---------------------------------------------------------------------------------------------------
namespace {
struct SparseWorld {
    cv::Mat map;
    std::vector<ReferenceCrop> crops;
    int cx, cy;
};

const SparseWorld& sparseWorld()
{
    static SparseWorld w = [] {
        SparseWorld w;
        w.map = testutil::makeTexture(1200, 1200, 777);
        w.cx = w.map.cols / 2;
        w.cy = w.map.rows / 2;
        w.crops = generateReferenceCropsGrid(w.map, kLat, kLng, w.cx, w.cy, kMpp, kMpdLat, kMpdLng,
                                             /*spacing*/ 200, /*crop*/ kReferenceCropMeters);
        return w;
    }();
    return w;
}

cv::Mat viewOf(const cv::Mat& map, const cv::Point2f& centre_px, double angle_deg)
{
    const int take = static_cast<int>(kReferenceCropMeters / kMpp);
    cv::Mat M = cv::getRotationMatrix2D(centre_px, angle_deg, 1.0);
    M.at<double>(0, 2) += take / 2.0 - centre_px.x;
    M.at<double>(1, 2) += take / 2.0 - centre_px.y;
    cv::Mat out;
    cv::warpAffine(map, out, M, cv::Size(take, take), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return out;
}

std::unique_ptr<SplitPipelineEstimator> makeSparseEstimator(bool recentre)
{
    const auto& w = sparseWorld();
    auto est = std::make_unique<SplitPipelineEstimator>(
        std::make_unique<GlobalHistogramRetrieval>(), std::make_unique<OrbRansacMatching>(1000),
        "test_split_rc", static_cast<int>(w.crops.size()), /*threads*/ 1);
    est->precompute(w.crops);
    est->setGeoReference(kMpp, kMpdLat, kMpdLng);
    if (recentre) {
        EXPECT_TRUE(est->enableCandidateRecentring(w.map, kLat, kLng, w.cx, w.cy, /*passes*/ 1));
    }
    return est;
}

double sparseErrorMeters(const PositionEstimate& e, const cv::Point2f& truth_px)
{
    const auto& w = sparseWorld();
    auto truth = CoordinateUtils::pixelToLatLng(static_cast<int>(truth_px.x), static_cast<int>(truth_px.y), kLat,
                                                kLng, w.cx, w.cy, kMpp, kMpdLat, kMpdLng);
    return CoordinateUtils::calculateDistance(truth, e.position, kMpdLat, kMpdLng);
}

// A matcher with no uncached-reference path (the IMatchingStage default).
class CacheOnlyMatcher : public IMatchingStage {
public:
    MatchResult match(const cv::Mat&, const ReferenceCrop&) override { return {}; }
};
}  // namespace

// Grid points sit at multiples of 200 px from the map edge; (460, 660) is ~85 m from the nearest one,
// so the best grid crop overlaps the view by only ~64%. Re-centred, it should overlap almost fully.
TEST(SplitPipelineRecentring, RaisesInliersOnAMisCentredViewAndKeepsThePosition)
{
    cv::setNumThreads(0);
    const cv::Point2f truth(460.0f, 660.0f);
    const cv::Mat view = viewOf(sparseWorld().map, truth, 8.0);
    PositionEstimate base = makeSparseEstimator(false)->estimatePosition(view, sparseWorld().crops);
    PositionEstimate rc   = makeSparseEstimator(true)->estimatePosition(view, sparseWorld().crops);
    ASSERT_TRUE(base.measurement_valid);
    ASSERT_TRUE(rc.measurement_valid);
    EXPECT_GT(rc.inliers, base.inliers * 1.2) << "base " << base.inliers << " re-centred " << rc.inliers;
    EXPECT_LT(sparseErrorMeters(rc, truth), 5.0);
    // Telemetry stays on the GRID (recall@k keeps its meaning): best_match_idx is a grid index and every
    // candidate carries a grid crop's exact coordinates, not its re-centred crop's.
    // (The winner may be a NEIGHBOURING cell that re-centred onto the same spot -- legitimate.)
    ASSERT_GE(rc.best_match_idx, 0);
    ASSERT_LT(rc.best_match_idx, static_cast<int>(sparseWorld().crops.size()));
    ASSERT_FALSE(rc.candidates.empty());
    for (const auto& cand : rc.candidates) {
        bool on_grid = false;
        for (const auto& c : sparseWorld().crops) on_grid = on_grid || c.coordinates == cand.first;
        EXPECT_TRUE(on_grid);
    }
}

// Each frame's re-centred crops are fresh buffers that routinely reuse the previous ones' addresses;
// a frame reusing the same frame buffer must still be localised from its own content.
TEST(SplitPipelineRecentring, ReusedBuffersAreNotAStaleCache)
{
    cv::setNumThreads(0);
    auto est = makeSparseEstimator(true);
    cv::Mat buffer(300, 300, CV_8UC3);
    const uchar* addr = buffer.data;
    for (const cv::Point2f truth : {cv::Point2f(460, 660), cv::Point2f(735, 330), cv::Point2f(530, 880)}) {
        viewOf(sparseWorld().map, truth, -15.0).copyTo(buffer);
        ASSERT_EQ(buffer.data, addr);
        PositionEstimate e = est->estimatePosition(buffer, sparseWorld().crops);
        ASSERT_TRUE(e.measurement_valid);
        EXPECT_LT(sparseErrorMeters(e, truth), 5.0) << "truth px " << truth;
    }
}

TEST(SplitPipelineRecentring, RefusedWithoutAnUncachedMatcherOrWithARefiner)
{
    const auto& w = sparseWorld();
    SplitPipelineEstimator cache_only(std::make_unique<GlobalHistogramRetrieval>(),
                                      std::make_unique<CacheOnlyMatcher>(), "cache_only");
    cache_only.setGeoReference(kMpp, kMpdLat, kMpdLng);
    EXPECT_FALSE(cache_only.enableCandidateRecentring(w.map, kLat, kLng, w.cx, w.cy, 1));

    SplitPipelineEstimator refined(std::make_unique<GlobalHistogramRetrieval>(),
                                   std::make_unique<OrbRansacMatching>(1000), "refined", 10, 1,
                                   std::make_unique<OrbRansacMatching>(1000));
    EXPECT_FALSE(refined.enableCandidateRecentring(w.map, kLat, kLng, w.cx, w.cy, 1));

    SplitPipelineEstimator ok(std::make_unique<GlobalHistogramRetrieval>(),
                              std::make_unique<OrbRansacMatching>(1000), "ok");
    EXPECT_FALSE(ok.enableCandidateRecentring(w.map, kLat, kLng, w.cx, w.cy, 0));  // 0 passes = off
    EXPECT_FALSE(ok.enableCandidateRecentring(cv::Mat(), kLat, kLng, w.cx, w.cy, 1));
    EXPECT_TRUE(ok.enableCandidateRecentring(w.map, kLat, kLng, w.cx, w.cy, 1));
}

// The matcher-level contract: matchUncachedReference() on a crop buffer that is overwritten in place
// (same cv::Mat::data, new content) must follow the NEW content every time.
TEST(SplitPipelineRecentring, OrbUncachedReferenceFollowsTheBuffersCurrentContent)
{
    cv::setNumThreads(0);
    const cv::Mat& map = sparseWorld().map;
    OrbRansacMatching matcher(1000);
    const cv::Mat frame = map(cv::Rect(450, 450, 300, 300)).clone();
    cv::Mat crop_buffer(300, 300, CV_8UC3);
    const uchar* addr = crop_buffer.data;
    for (const cv::Point origin : {cv::Point(420, 470), cv::Point(480, 430), cv::Point(465, 455)}) {
        map(cv::Rect(origin.x, origin.y, 300, 300)).copyTo(crop_buffer);
        ASSERT_EQ(crop_buffer.data, addr);
        MatchResult m = matcher.matchUncachedReference(frame, ReferenceCrop({0.0, 0.0}, crop_buffer));
        ASSERT_FALSE(m.homography.empty()) << "origin " << origin;
        // frame px p -> map px p + (450, 450) -> crop px p + (450, 450) - origin
        EXPECT_NEAR(m.homography.at<double>(0, 2), 450 - origin.x, 2.0) << "origin " << origin;
        EXPECT_NEAR(m.homography.at<double>(1, 2), 450 - origin.y, 2.0) << "origin " << origin;
    }
}
