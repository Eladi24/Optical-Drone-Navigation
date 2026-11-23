// CoordinateUtils: planar lat/lng <-> pixel. At ~32 N a degree of longitude is only ~85% of
// a degree of latitude, so mixing up the two factors shifts positions by hundreds of metres
// (the generateReferenceCropsGrid axis swap, the Kalman "111320 for lng" bug).
#include "CoordinateUtils.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>

namespace {
constexpr double kLat = 32.0, kLng = 35.0;
const double kMpdLat = testutil::kMetersPerDegLat;
const double kMpdLng = testutil::metersPerDegLng(kLat);
constexpr int kCx = 1000, kCy = 800;
constexpr double kMpp = 1.2;
}  // namespace

TEST(CoordinateUtils, LatLngPixelRoundTripWithAnisotropicDegrees)
{
    ASSERT_LT(kMpdLng / kMpdLat, 0.86);  // the anisotropy is real at this latitude
    cv::RNG rng(7);
    for (int i = 0; i < 200; ++i) {
        const double lat = kLat + rng.uniform(-0.007, 0.007);
        const double lng = kLng + rng.uniform(-0.009, 0.009);
        cv::Point px = CoordinateUtils::latLngToPixel(lat, lng, kLat, kLng, kCx, kCy, kMpp, kMpdLat, kMpdLng);
        auto back = CoordinateUtils::pixelToLatLng(px.x, px.y, kLat, kLng, kCx, kCy, kMpp, kMpdLat, kMpdLng);
        const double err_m = CoordinateUtils::calculateDistance({lat, lng}, back, kMpdLat, kMpdLng);
        EXPECT_LT(err_m, 1.5 * kMpp * std::sqrt(2.0));  // int truncation: under ~1 px per axis
    }
}

TEST(CoordinateUtils, EastStepUsesLongitudeFactor)
{
    const double d_lng = 0.01;
    cv::Point px = CoordinateUtils::latLngToPixel(kLat, kLng + d_lng, kLat, kLng, kCx, kCy, kMpp, kMpdLat, kMpdLng);
    EXPECT_NEAR(px.x - kCx, d_lng * kMpdLng / kMpp, 1.0);   // ~787 px, not ~928 (lat factor)
    EXPECT_EQ(px.y, kCy);
    EXPECT_NEAR(CoordinateUtils::calculateDistance({kLat, kLng}, {kLat, kLng + d_lng}, kMpdLat, kMpdLng),
                d_lng * kMpdLng, 1e-6);
}

TEST(CoordinateUtils, NorthIsImageUp)
{
    cv::Point px = CoordinateUtils::latLngToPixel(kLat + 0.001, kLng, kLat, kLng, kCx, kCy, kMpp, kMpdLat, kMpdLng);
    EXPECT_EQ(px.x, kCx);
    EXPECT_LT(px.y, kCy);
    EXPECT_NEAR(kCy - px.y, 0.001 * kMpdLat / kMpp, 1.0);
}

TEST(CoordinateUtils, HeadingIsClockwiseFromNorth)
{
    const std::pair<double, double> o{kLat, kLng};
    EXPECT_NEAR(CoordinateUtils::calculateHeading(o, {kLat + 0.001, kLng}, kMpdLat, kMpdLng), 0.0, 1e-9);
    EXPECT_NEAR(CoordinateUtils::calculateHeading(o, {kLat, kLng + 0.001}, kMpdLat, kMpdLng), 90.0, 1e-9);
    EXPECT_NEAR(CoordinateUtils::calculateHeading(o, {kLat - 0.001, kLng}, kMpdLat, kMpdLng), 180.0, 1e-9);
    EXPECT_NEAR(CoordinateUtils::calculateHeading(o, {kLat, kLng - 0.001}, kMpdLat, kMpdLng), 270.0, 1e-9);
    // Equal metric steps north and east -> exactly 45 deg (needs the per-axis factors).
    const double step_m = 100.0;
    EXPECT_NEAR(CoordinateUtils::calculateHeading(o, {kLat + step_m / kMpdLat, kLng + step_m / kMpdLng},
                                                  kMpdLat, kMpdLng), 45.0, 1e-9);
}
