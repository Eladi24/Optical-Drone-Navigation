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
    EXPECT_EQ(view.layout(), cv::Size(3, 1));                // square panels fit the box best side by side
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
    const cv::Point o0 = view.panelOrigin(0), d1 = view.panelOrigin(1) - o0, d2 = view.panelOrigin(2) - o0;

    // A point in the middle of the first truth segment and of the first estimate segment (away from the
    // crosshair, which runs through the LAST truth point).
    const cv::Point2d g = view.toPanel(0.40, 0.40), e = view.toPanel(0.45, 0.40);
    const cv::Point gp = o0 + cv::Point(static_cast<int>(std::lround(g.x)), static_cast<int>(std::lround(g.y)));
    const cv::Point ep = o0 + cv::Point(static_cast<int>(std::lround(e.x)), static_cast<int>(std::lround(e.y)));
    const cv::Vec3b white(255, 255, 255), magenta(255, 0, 255);

    EXPECT_TRUE(isColour(out, gp, white));                          // truth panel: truth...
    EXPECT_FALSE(isColour(out, ep, magenta));                       // ...and no estimate
    EXPECT_TRUE(isColour(out, ep + d1, magenta));   // estimate panel: estimate...
    EXPECT_FALSE(isColour(out, gp + d1, white));    // ...and no truth
    EXPECT_TRUE(isColour(out, gp + d2, white));     // overlay: both
    EXPECT_TRUE(isColour(out, ep + d2, magenta));
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

// 2026-11-26: a wide flight (AerialVL av2's crop is ~2:1) gave a 6:1 strip with the panels side by side -- a
// short, very wide window. The grid is now chosen to fill a screen-shaped box.
TEST(NavigationView, LayoutFollowsThePanelShape)
{
    EXPECT_EQ(NavigationView::chooseLayout(3, {1872, 900}), cv::Size(2, 2));   // av2-like, 2.08:1
    EXPECT_EQ(NavigationView::chooseLayout(3, {450, 900}), cv::Size(3, 1));    // av1-like, 1:2
    EXPECT_EQ(NavigationView::chooseLayout(3, {270, 900}), cv::Size(3, 1));    // flight 01-like, 0.3:1
    EXPECT_EQ(NavigationView::chooseLayout(3, {5000, 900}), cv::Size(1, 3));   // extreme east-west strip
    EXPECT_EQ(NavigationView::chooseLayout(1, {5000, 900}), cv::Size(1, 1));   // no ground truth: one panel

    // Rendered: a wide ground-truth path -> 2x2 grid, panel 2 (overlay) in the second row, first column.
    const cv::Mat map = testutil::makeTexture(1200, 1000, 8);
    const std::vector<LL> gt = {{0.40, 0.10}, {0.60, 0.50}};   // px (100,400) .. (500,600): crop 441 x 241, ~1.8:1
    const NavigationView view(map, toyProjection(), &gt, 1.0, 300, 20.0);
    ASSERT_EQ(view.layout(), cv::Size(2, 2));
    const cv::Size p = view.panelSize();
    const cv::Mat out = view.render(gt, gt, gt, gt.front());
    EXPECT_EQ(out.cols, 2 * p.width + NavigationView::kGap);
    EXPECT_EQ(out.rows, NavigationView::kLegend + 2 * (NavigationView::kTitle + p.height) + NavigationView::kGap);
    EXPECT_EQ(view.panelOrigin(1), cv::Point(p.width + NavigationView::kGap, NavigationView::kHeader));
    EXPECT_EQ(view.panelOrigin(2), cv::Point(0, NavigationView::kHeader + p.height + NavigationView::kGap +
                                                NavigationView::kTitle));
    EXPECT_DOUBLE_EQ(view.uiScale(), 1.0);   // a small grid keeps the base header sizes

    // A production-size wide view (panels 900 px tall, grid ~2x the on-screen width): header bands scale up so the
    // titles/legend stay readable once the window shrinks the image, and the panel origins follow them.
    const NavigationView big(map, toyProjection(), &gt, 1.0, 900, 20.0);
    ASSERT_EQ(big.layout(), cv::Size(2, 2));
    EXPECT_GT(big.uiScale(), 1.5);
    EXPECT_EQ(big.panelOrigin(0).y, big.legendHeight() + big.titleHeight());
    const cv::Mat bout = big.render(gt, gt, gt, gt.front());
    EXPECT_EQ(bout.rows, big.legendHeight() + 2 * (big.titleHeight() + big.panelSize().height) + NavigationView::kGap);
}
