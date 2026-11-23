// CameraModel: altitude <-> footprint <-> GSD. The GSD-normalised crop makes this math
// load-bearing: a wrong HFOV silently matches two different physical scales.
#include "CameraModel.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace CameraModel;

TEST(CameraModel, FootprintMatchesHandComputedValues)
{
    // 90 deg HFOV: half-angle 45 deg, tan = 1 -> footprint = 2 * altitude.
    EXPECT_NEAR(frameFootprintMeters(100.0, 90.0), 200.0, 1e-9);
    // 60 deg HFOV: tan(30) = 1/sqrt(3) -> footprint = 2 * 300 / sqrt(3).
    EXPECT_NEAR(frameFootprintMeters(300.0, 60.0), 600.0 / std::sqrt(3.0), 1e-9);
}

TEST(CameraModel, AltitudeFootprintRoundTrip)
{
    for (double alt : {50.0, 250.0, 550.0, 1200.0})
        for (double hfov : {kUavVisLocHfovDeg, kDefaultHfovDeg})
            EXPECT_NEAR(altitudeFromFootprintMeters(frameFootprintMeters(alt, hfov), hfov), alt, 1e-9);
}

TEST(CameraModel, GsdIsFootprintOverWidth)
{
    EXPECT_NEAR(groundSampleDistance(100.0, 2000, 90.0), 0.1, 1e-12);
}

// Regression for the 2026-11-16 HFOV bug: UAV-VisLoc's camera is 51.5 deg (measured from
// still-to-still registration vs GPS), not the 70 deg default. Using 70 overstated GSD by
// ~1.45x on every flight. Pin both the constant and the size of the error it guards against.
TEST(CameraModel, UavVisLocHfovIsTheMeasuredValue)
{
    EXPECT_DOUBLE_EQ(kUavVisLocHfovDeg, 51.5);
    EXPECT_DOUBLE_EQ(kDefaultHfovDeg, 70.0);
    const double ratio = groundSampleDistance(500.0, 4000, kDefaultHfovDeg) /
                         groundSampleDistance(500.0, 4000, kUavVisLocHfovDeg);
    EXPECT_NEAR(ratio, 1.45, 0.01);
}
