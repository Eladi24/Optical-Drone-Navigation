// NavigationView (the live three-panel map, 2026-11-24): panel geometry, the crop around the ground truth,
// and that each panel draws only what its title says (truth panel: truth; estimate panel: estimate).
#include "NavigationView.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>

namespace {
using LL = NavigationView::LatLng;

// Toy projection: lat/lng are already map pixels / 1000 (y = lat * 1000, x = lng * 1000).
NavigationView::Project toyProjection()
{
    return [](double lat, double lng) {
        return cv::Point(static_cast<int>(std::lround(lng * 1000)), static_cast<int>(std::lround(lat * 1000)));
    };
}

bool isColour(const cv::Mat& img, cv::Point p, cv::Vec3b c) { return img.at<cv::Vec3b>(p) == c; }
}  // namespace

TEST(NavigationView, ThreePanelsCroppedAroundTheGroundTruth)
{
    const cv::Mat map = testutil::makeTexture(1000, 1000, 5);
    const std::vector<LL> gt = {{0.40, 0.30}, {0.40, 0.50}, {0.60, 0.50}};   // px (300,400) .. (500,600)
    const NavigationView view(map, toyProjection(), &gt, /*mpp*/ 1.0, /*panel_height*/ 400, /*margin_m*/ 50.0);

    EXPECT_EQ(view.panelCount(), 3);
    EXPECT_EQ(view.crop(), cv::Rect(250, 350, 301, 301));   // bbox (300..500, 400..600) +- 50, inclusive
    const cv::Mat out = view.render(gt, gt, gt, gt.front());
    const cv::Size p = view.panelSize();
    EXPECT_EQ(p.height, 400);
    EXPECT_EQ(out.rows, NavigationView::kHeader + p.height);
    EXPECT_EQ(out.cols, 3 * p.width + 2 * NavigationView::kGap);
    const cv::Point2d c = view.toPanel(0.40, 0.30);   // the first GT point, 50 px inside the crop corner
    EXPECT_NEAR(c.x, 50 * 400.0 / 301, 1.0);
    EXPECT_NEAR(c.y, 50 * 400.0 / 301, 1.0);
}

TEST(NavigationView, EachPanelDrawsOnlyItsOwnPath)
{
    const cv::Mat map = testutil::makeTexture(1000, 1000, 6);
    const std::vector<LL> gt  = {{0.40, 0.30}, {0.40, 0.50}, {0.60, 0.50}};
    const std::vector<LL> est = {{0.45, 0.30}, {0.45, 0.50}, {0.65, 0.50}};   // 50 px south of the truth
    const NavigationView view(map, toyProjection(), &gt, 1.0, 400, 100.0);
    const cv::Mat out = view.render(gt, est, est, {0.0, 0.0});
    const int pw = view.panelSize().width, step = pw + NavigationView::kGap, hdr = NavigationView::kHeader;

    // A point in the middle of the first truth segment and of the first estimate segment (away from the
    // crosshair, which runs through the LAST truth point).
    const cv::Point2d g = view.toPanel(0.40, 0.40), e = view.toPanel(0.45, 0.40);
    const cv::Point gp(static_cast<int>(std::lround(g.x)), hdr + static_cast<int>(std::lround(g.y)));
    const cv::Point ep(static_cast<int>(std::lround(e.x)), hdr + static_cast<int>(std::lround(e.y)));
    const cv::Vec3b white(255, 255, 255), magenta(255, 0, 255);

    EXPECT_TRUE(isColour(out, gp, white));                          // truth panel: truth...
    EXPECT_FALSE(isColour(out, ep, magenta));                       // ...and no estimate
    EXPECT_TRUE(isColour(out, ep + cv::Point(step, 0), magenta));   // estimate panel: estimate...
    EXPECT_FALSE(isColour(out, gp + cv::Point(step, 0), white));    // ...and no truth
    EXPECT_TRUE(isColour(out, gp + cv::Point(2 * step, 0), white)); // overlay: both
    EXPECT_TRUE(isColour(out, ep + cv::Point(2 * step, 0), magenta));
}

TEST(NavigationView, WithoutGroundTruthOneFullMapPanel)
{
    const cv::Mat map = testutil::makeTexture(800, 600, 7);
    const NavigationView view(map, toyProjection(), nullptr, 1.0, 300);
    EXPECT_EQ(view.panelCount(), 1);
    EXPECT_EQ(view.crop(), cv::Rect(0, 0, 800, 600));
    const cv::Mat out = view.render({}, {}, {{0.2, 0.2}, {0.3, 0.4}}, {0.0, 0.0});
    EXPECT_EQ(out.cols, view.panelSize().width);
    EXPECT_EQ(out.rows, NavigationView::kHeader + 300);
}
