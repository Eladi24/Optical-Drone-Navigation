#pragma once
#include <opencv2/core.hpp>
#include <functional>
#include <utility>
#include <vector>

// The live "Video Navigation" map (2026-11-24, user request): the overlaid orange ground truth and
// confidence-coloured estimate were hard to tell apart, so the same dimmed, cropped map region is shown
// in three synchronised panels:
//     [ GROUND TRUTH ]  [ ESTIMATE (filtered line + raw-match dots) ]  [ OVERLAY + ERROR WHISKERS ]
// Whiskers join each frame's estimate to that frame's true position, coloured by error (< 50 m, < 150 m,
// worse), so where and how badly the estimate is off is visible at a glance. A thin crosshair at the true
// current position is drawn in every panel so the eye can compare across them. The crop is the whole
// ground-truth path's bounding box plus a margin, fixed from frame 0. Without ground truth (e.g. Haifa
// Sample 3) there is a single panel: the full dimmed map with the estimate only.
// Drawing only: nothing here feeds back into estimation or the telemetry CSV.
class NavigationView {
public:
    using LatLng  = std::pair<double, double>;
    using Project = std::function<cv::Point(double lat, double lng)>;   // lat/lng -> reference-map pixel

    NavigationView(const cv::Mat& map, Project to_map_px, const std::vector<LatLng>* full_gt_path,
                   double map_mpp, int panel_height = 900, double margin_m = 300.0);

    // gt_drawn / raw / filtered: one entry per processed frame, {0,0} = no value (processVideoNavigation's
    // convention). Returns header strip + panels side by side.
    cv::Mat render(const std::vector<LatLng>& gt_drawn, const std::vector<LatLng>& raw,
                   const std::vector<LatLng>& filtered, const LatLng& start) const;

    int panelCount() const { return has_gt_ ? 3 : 1; }
    cv::Size panelSize() const { return base_.size(); }
    const cv::Rect& crop() const { return crop_; }
    cv::Point2d toPanel(double lat, double lng) const;   // the same pixel in every panel

    static constexpr int kGap    = 8;    // px between panels
    static constexpr int kHeader = 70;   // title row + legend row above the panels

private:
    Project  to_px_;
    double   mpp_;
    bool     has_gt_ = false;
    cv::Rect crop_;
    double   scale_ = 1.0;   // panel px per map px
    cv::Mat  base_;          // one panel's background: cropped, greyed, dimmed, resized map
};
