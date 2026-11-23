// DroneKalmanFilter (4-state constant velocity in degrees). Regressions for the Kalman bugs
// found in the STRATEGY.md code review: velocity converted with the latitude factor for
// longitude ("111320 for lng"), and the innovation gate.
#include "KalmanFilter.hpp"
#include "CoordinateUtils.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>

namespace {
constexpr double kLat = 32.0, kLng = 35.0;
const double kMpdLat = testutil::kMetersPerDegLat;
const double kMpdLng = testutil::metersPerDegLng(kLat);
}  // namespace

TEST(Kalman, EastwardVelocityUsesLongitudeFactor)
{
    DroneKalmanFilter kf(1.0, 20.0, kMpdLat, kMpdLng);
    kf.initialize(kLat, kLng, /*heading*/ 90.0, /*speed m/s*/ 10.0);
    auto p = kf.predict(10.0);
    EXPECT_NEAR((p.second - kLng) * kMpdLng, 100.0, 0.5);  // the bug gave ~84.8 m at 32 N
    EXPECT_NEAR((p.first - kLat) * kMpdLat, 0.0, 0.5);
}

TEST(Kalman, NorthwardVelocity)
{
    DroneKalmanFilter kf(1.0, 20.0, kMpdLat, kMpdLng);
    kf.initialize(kLat, kLng, 0.0, 10.0);
    auto p = kf.predict(10.0);
    EXPECT_NEAR((p.first - kLat) * kMpdLat, 100.0, 0.5);
    EXPECT_NEAR((p.second - kLng) * kMpdLng, 0.0, 0.5);
}

TEST(Kalman, LowConfidenceOutlierIsRejected)
{
    DroneKalmanFilter kf(1.0, 20.0, kMpdLat, kMpdLng);
    kf.initialize(kLat, kLng, 0.0, 0.0);
    kf.predict(1.0);
    const double far_lng = kLng + 500.0 / kMpdLng;  // 500 m east, beyond the 50 m gate
    auto p = kf.update(kLat, far_lng, /*confidence*/ 0.3, /*innovation_threshold*/ 50.0);
    EXPECT_LT(CoordinateUtils::calculateDistance({kLat, kLng}, p, kMpdLat, kMpdLng), 1.0);
}

TEST(Kalman, HighConfidenceMeasurementIsFused)
{
    DroneKalmanFilter kf(1.0, 20.0, kMpdLat, kMpdLng);
    kf.initialize(kLat, kLng, 0.0, 0.0);
    kf.predict(1.0);
    const double far_lng = kLng + 500.0 / kMpdLng;
    auto p = kf.update(kLat, far_lng, 0.9, 50.0);
    EXPECT_GT(CoordinateUtils::calculateDistance({kLat, kLng}, p, kMpdLat, kMpdLng), 10.0);
}

TEST(Kalman, TracksAConstantVelocityTarget)
{
    DroneKalmanFilter kf(1.0, 20.0, kMpdLat, kMpdLng);
    kf.initialize(kLat, kLng, 90.0, 0.0);  // starts with the wrong (zero) speed
    cv::RNG rng(3);
    std::pair<double, double> est{kLat, kLng}, truth{kLat, kLng};
    for (int t = 1; t <= 60; ++t) {
        truth = {kLat, kLng + 10.0 * t / kMpdLng};  // 10 m/s east
        kf.predict(1.0);
        est = kf.update(truth.first + rng.gaussian(10.0) / kMpdLat,
                        truth.second + rng.gaussian(10.0) / kMpdLng, 0.8, 50.0);
    }
    EXPECT_LT(CoordinateUtils::calculateDistance(truth, est, kMpdLat, kMpdLng), 15.0);
}
