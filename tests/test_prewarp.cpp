// Compass heading pre-warp. Regression for the 2026-11-18 sign bug: north-up needs a rotation
// by -heading (UAV-VisLoc Phi1, degrees clockwise from north); the old +heading rotated frames
// AWAY from north-up, which made DISK+LightGlue (not rotation-invariant) fail on every leg of a
// lawnmower survey that wasn't flown north.
#include "VideoProcessing.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace {
// A camera with heading H (image-up points H degrees clockwise from north) sees world-north
// rotated counter-clockwise by H in the image: H=0 -> up, H=90 -> left, H=180 -> down.
cv::Point2f northMarkerInImage(const cv::Point2f& centre, double r, double heading_deg)
{
    const double h = heading_deg * M_PI / 180.0;
    return {static_cast<float>(centre.x - r * std::sin(h)), static_cast<float>(centre.y - r * std::cos(h))};
}

cv::Point2f brightCentroid(const cv::Mat& gray)
{
    cv::Moments m = cv::moments(gray > 127, true);
    return {static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00)};
}
}  // namespace

TEST(HeadingPrewarp, TurnsTheFrameNorthUp)
{
    const int take = 240;
    const double r = 80.0;
    const cv::Point2f centre(260.0f, 190.0f);  // off-centre crop: exercises the translation too
    for (double heading : {0.0, 30.0, 90.0, 135.0, 180.0, 250.0, -60.0}) {
        cv::Mat frame(400, 560, CV_8UC1, cv::Scalar(0));
        cv::circle(frame, northMarkerInImage(centre, r, heading), 6, cv::Scalar(255), cv::FILLED);
        cv::Mat out = headingPrewarpCrop(frame, centre, take, heading);
        ASSERT_EQ(out.size(), cv::Size(take, take));
        cv::Point2f p = brightCentroid(out);
        EXPECT_NEAR(p.x, take / 2.0, 1.5) << "heading " << heading;
        EXPECT_NEAR(p.y, take / 2.0 - r, 1.5) << "heading " << heading;  // north = straight up
    }
}

TEST(HeadingPrewarp, ZeroHeadingIsAPlainCrop)
{
    cv::Mat frame(300, 400, CV_8UC1);
    cv::randu(frame, 0, 256);
    const cv::Point2f centre(200.0f, 150.0f);
    cv::Mat out = headingPrewarpCrop(frame, centre, 100, 0.0);
    cv::Mat expected = frame(cv::Rect(150, 100, 100, 100));
    EXPECT_EQ(cv::norm(out, expected, cv::NORM_INF), 0.0);
}
