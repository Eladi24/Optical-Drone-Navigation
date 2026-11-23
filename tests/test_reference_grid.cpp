// generateReferenceCropsGrid. Regression for the axis-swap bug: map width (a longitude extent)
// was converted with the LATITUDE factor, so at ~32 N the grid covered only ~85% of the map's
// east/west extent (~686 m missing per side on an 8.8 km UAV-VisLoc map).
#include "VideoProcessing.hpp"
#include "CoordinateUtils.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>
#include <algorithm>

namespace {
constexpr double kLat = 32.0, kLng = 120.0;
constexpr double kMpp = 4.0;              // coarse, keeps memory small
constexpr int kSpacingM = 200, kCropM = 300;
const double kMpdLat = testutil::kMetersPerDegLat;
const double kMpdLng = testutil::metersPerDegLng(kLat);

struct Grid {
    cv::Mat map;
    std::vector<ReferenceCrop> crops;
};

Grid makeGrid()
{
    Grid g;
    g.map = cv::Mat(7000 / kMpp, 8800 / kMpp, CV_8UC1, cv::Scalar(128));  // 8.8 km x 7.0 km
    g.crops = generateReferenceCropsGrid(g.map, kLat, kLng, g.map.cols / 2, g.map.rows / 2,
                                         kMpp, kMpdLat, kMpdLng, kSpacingM, kCropM);
    return g;
}
}  // namespace

TEST(ReferenceGrid, CoversTheFullEastWestExtent)
{
    Grid g = makeGrid();
    ASSERT_FALSE(g.crops.empty());
    auto [mn, mx] = std::minmax_element(g.crops.begin(), g.crops.end(), [](const auto& a, const auto& b) {
        return a.coordinates.second < b.coordinates.second;
    });
    const double ew_extent_m = (mx->coordinates.second - mn->coordinates.second) * kMpdLng;
    const double map_width_m = g.map.cols * kMpp;
    // The bug gave ~0.85 * 8800 = ~7460 m; the correct grid spans the map to within a spacing per side.
    EXPECT_GT(ew_extent_m, map_width_m - 2 * kSpacingM);
    EXPECT_LT(ew_extent_m, map_width_m + kSpacingM);
}

TEST(ReferenceGrid, CoversTheFullNorthSouthExtent)
{
    Grid g = makeGrid();
    auto [mn, mx] = std::minmax_element(g.crops.begin(), g.crops.end(), [](const auto& a, const auto& b) {
        return a.coordinates.first < b.coordinates.first;
    });
    const double ns_extent_m = (mx->coordinates.first - mn->coordinates.first) * kMpdLat;
    const double map_height_m = g.map.rows * kMpp;
    EXPECT_GT(ns_extent_m, map_height_m - 2 * kSpacingM);
    EXPECT_LT(ns_extent_m, map_height_m + kSpacingM);
}

TEST(ReferenceGrid, CropCountAndSizesMatchTheSpacing)
{
    Grid g = makeGrid();
    const int crop_px = static_cast<int>(std::round(kCropM / kMpp));
    const double expected = (8800.0 / kSpacingM + 1) * (7000.0 / kSpacingM + 1);
    EXPECT_NEAR(static_cast<double>(g.crops.size()), expected, 0.1 * expected);
    for (const auto& c : g.crops) {
        EXPECT_LE(c.image.cols, crop_px);
        EXPECT_LE(c.image.rows, crop_px);
        EXPECT_GT(c.image.cols, crop_px / 2);  // edge crops may be clipped, but by at most half
        EXPECT_GT(c.image.rows, crop_px / 2);
    }
}

TEST(ReferenceGrid, CropCentresMapBackToTheirCoordinates)
{
    Grid g = makeGrid();
    for (const auto& c : g.crops) {
        cv::Point px = CoordinateUtils::latLngToPixel(c.coordinates.first, c.coordinates.second, kLat, kLng,
                                                      g.map.cols / 2, g.map.rows / 2, kMpp, kMpdLat, kMpdLng);
        // Grid points lie on the map (crops are clipped around them, never centred off-map by > half a crop).
        EXPECT_GE(px.x, -1);
        EXPECT_GE(px.y, -1);
        EXPECT_LE(px.x, g.map.cols);
        EXPECT_LE(px.y, g.map.rows);
    }
}
