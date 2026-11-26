#include "NavigationView.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
bool valid(const NavigationView::LatLng& p) { return p.first != 0.0 || p.second != 0.0; }

const cv::Scalar kTruth(255, 255, 255);     // white
const cv::Scalar kEstimate(255, 0, 255);    // magenta
const cv::Scalar kRawDot(230, 200, 60);     // muted cyan-blue
const cv::Scalar kCross(170, 170, 170);     // crosshair grey
const cv::Scalar kStart(203, 105, 255);     // pink
const cv::Scalar kGood(0, 200, 0), kMid(0, 165, 255), kBad(0, 0, 255);
const cv::Scalar kChrome(38, 38, 38);       // header + gaps

cv::Scalar errorColour(double m) { return m < 50.0 ? kGood : (m < 150.0 ? kMid : kBad); }
}  // namespace

NavigationView::NavigationView(const cv::Mat& map, Project to_map_px, const std::vector<LatLng>* full_gt_path,
                               double map_mpp, int panel_height, double margin_m)
    : to_px_(std::move(to_map_px)), mpp_(map_mpp)
{
    crop_ = cv::Rect(0, 0, map.cols, map.rows);
    if (full_gt_path) {
        int x0 = std::numeric_limits<int>::max(), y0 = x0, x1 = std::numeric_limits<int>::min(), y1 = x1, n = 0;
        for (const auto& g : *full_gt_path) {
            if (!valid(g)) continue;
            const cv::Point p = to_px_(g.first, g.second);
            x0 = std::min(x0, p.x); y0 = std::min(y0, p.y); x1 = std::max(x1, p.x); y1 = std::max(y1, p.y);
            ++n;
        }
        if (n >= 2) {
            const int m = mpp_ > 0.0 ? static_cast<int>(std::lround(margin_m / mpp_)) : 0;
            const cv::Rect box = cv::Rect(cv::Point(x0 - m, y0 - m), cv::Point(x1 + m + 1, y1 + m + 1)) &
                                 cv::Rect(0, 0, map.cols, map.rows);
            if (box.width > 1 && box.height > 1) {
                crop_ = box;
                has_gt_ = true;
            }
        }
    }
    cv::Mat grey, dim;
    cv::cvtColor(map(crop_), grey, cv::COLOR_BGR2GRAY);
    cv::cvtColor(grey, dim, cv::COLOR_GRAY2BGR);
    dim.convertTo(dim, -1, 0.55, 0.0);   // dimmed + greyscale so the coloured paths stand out
    scale_ = static_cast<double>(panel_height) / crop_.height;
    cv::resize(dim, base_, cv::Size(std::max(1, static_cast<int>(std::lround(crop_.width * scale_))), panel_height),
               0, 0, cv::INTER_AREA);
    layout_ = chooseLayout(panelCount(), base_.size());
    ui_ = std::max(1.0, (layout_.width * base_.cols + (layout_.width - 1) * kGap) / 1800.0);
}

cv::Size NavigationView::chooseLayout(int k, cv::Size panel)
{
    const std::vector<cv::Size> candidates = k == 3 ? std::vector<cv::Size>{{3, 1}, {2, 2}, {1, 3}}
                                                    : std::vector<cv::Size>{{1, 1}};
    cv::Size best = candidates.front();
    double best_scale = -1.0;
    for (const cv::Size& g : candidates) {   // in order of increasing rows: a tie keeps the flatter grid
        const double w = g.width * panel.width + (g.width - 1) * kGap;
        const double h = kLegend + g.height * (kTitle + panel.height) + (g.height - 1) * kGap;
        const double scale = std::min(kFitW / w, kFitH / h);
        if (scale > best_scale + 1e-9) { best_scale = scale; best = g; }
    }
    return best;
}

cv::Point NavigationView::panelOrigin(int i) const
{
    const int c = i % layout_.width, r = i / layout_.width;
    return {c * (base_.cols + kGap), legendHeight() + titleHeight() + r * (titleHeight() + base_.rows + kGap)};
}

cv::Point2d NavigationView::toPanel(double lat, double lng) const
{
    const cv::Point p = to_px_(lat, lng);
    return {(p.x - crop_.x) * scale_, (p.y - crop_.y) * scale_};
}

cv::Mat NavigationView::render(const std::vector<LatLng>& gt_drawn, const std::vector<LatLng>& raw,
                               const std::vector<LatLng>& filtered, const LatLng& start) const
{
    auto pt = [&](const LatLng& ll) {
        const cv::Point2d p = toPanel(ll.first, ll.second);
        return cv::Point(static_cast<int>(std::lround(p.x)), static_cast<int>(std::lround(p.y)));
    };
    auto drawPath = [&](cv::Mat& img, const std::vector<LatLng>& path, const cv::Scalar& c, int thick) {
        for (size_t i = 1; i < path.size(); ++i)
            if (valid(path[i - 1]) && valid(path[i])) cv::line(img, pt(path[i - 1]), pt(path[i]), c, thick, cv::LINE_AA);
    };
    auto lastValid = [](const std::vector<LatLng>& v) -> const LatLng* {
        for (auto it = v.rbegin(); it != v.rend(); ++it) if (valid(*it)) return &*it;
        return nullptr;
    };
    const LatLng* cur_gt  = lastValid(gt_drawn);
    const LatLng* cur_est = lastValid(filtered);
    const LatLng* cross   = cur_gt ? cur_gt : cur_est;

    auto decorate = [&](cv::Mat& img, bool show_truth_dot, bool show_est_dot) {
        if (cross) {
            const cv::Point c = pt(*cross);
            cv::line(img, {c.x, 0}, {c.x, img.rows - 1}, kCross, 1);
            cv::line(img, {0, c.y}, {img.cols - 1, c.y}, kCross, 1);
        }
        if (valid(start)) cv::circle(img, pt(start), 7, kStart, -1, cv::LINE_AA);
        if (show_truth_dot && cur_gt) {
            cv::circle(img, pt(*cur_gt), 7, kTruth, -1, cv::LINE_AA);
            cv::circle(img, pt(*cur_gt), 8, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
        if (show_est_dot && cur_est) {
            cv::circle(img, pt(*cur_est), 7, kEstimate, -1, cv::LINE_AA);
            cv::circle(img, pt(*cur_est), 8, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
    };

    cv::Mat est = base_.clone();
    for (const auto& r : raw) if (valid(r)) cv::circle(est, pt(r), 2, kRawDot, -1, cv::LINE_AA);
    drawPath(est, filtered, kEstimate, 2);
    decorate(est, false, true);

    std::vector<cv::Mat> panels;
    std::vector<std::string> titles;
    if (has_gt_) {
        cv::Mat truth = base_.clone();
        drawPath(truth, gt_drawn, kTruth, 3);
        decorate(truth, true, false);

        cv::Mat ov = base_.clone();
        const size_t n = std::min(gt_drawn.size(), filtered.size());
        for (size_t i = 0; i < n; ++i) {
            if (!valid(gt_drawn[i]) || !valid(filtered[i])) continue;
            const cv::Point a = pt(filtered[i]), b = pt(gt_drawn[i]);
            const double err_m = std::hypot(a.x - b.x, a.y - b.y) / scale_ * mpp_;
            cv::line(ov, a, b, errorColour(err_m), 1, cv::LINE_AA);
        }
        drawPath(ov, gt_drawn, kTruth, 2);
        drawPath(ov, filtered, kEstimate, 2);
        decorate(ov, true, true);

        panels = {truth, est, ov};
        titles = {"TRUTH", "ESTIMATE", "OVERLAY + ERROR"};
    } else {
        panels = {est};
        titles = {"ESTIMATE (no ground truth)"};
    }

    const int pw = base_.cols, ph = base_.rows, k = static_cast<int>(panels.size());
    const int gc = layout_.width, gr = layout_.height;
    const double u = ui_;
    cv::Mat out(legendHeight() + gr * (titleHeight() + ph) + (gr - 1) * kGap, gc * pw + (gc - 1) * kGap, base_.type(),
                kChrome);
    for (int i = 0; i < k; ++i) {
        const cv::Point o = panelOrigin(i);
        panels[i].copyTo(out(cv::Rect(o.x, o.y, pw, ph)));
        // Title in the band above the panel, shrunk to fit it (tall, narrow flights give narrow panels).
        double fs = 0.75 * u;
        const int th = std::max(2, static_cast<int>(std::lround(2 * u)));
        int base = 0;
        while (fs > 0.35 && cv::getTextSize(titles[i], cv::FONT_HERSHEY_SIMPLEX, fs, th, &base).width > pw - 16)
            fs -= 0.05;
        cv::putText(out, titles[i], {o.x + static_cast<int>(8 * u), o.y - static_cast<int>(10 * u)},
                    cv::FONT_HERSHEY_SIMPLEX, fs, cv::Scalar(235, 235, 235), th, cv::LINE_AA);
    }
    // Legend row, full width, once above the grid.
    auto U = [&](double v) { return static_cast<int>(std::lround(v * u)); };
    int x = U(10);
    const int y = U(24);
    const int lt = std::max(1, U(1));
    auto item = [&](const char* label, const cv::Scalar& col, bool dot) {
        if (dot) cv::circle(out, {x + U(10), y - U(5)}, U(4), col, -1, cv::LINE_AA);
        else     cv::line(out, {x, y - U(5)}, {x + U(22), y - U(5)}, col, U(3), cv::LINE_AA);
        cv::putText(out, label, {x + U(28), y}, cv::FONT_HERSHEY_SIMPLEX, 0.5 * u, cv::Scalar(215, 215, 215), lt,
                    cv::LINE_AA);
        int base = 0;
        x += U(40) + cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5 * u, lt, &base).width;
    };
    if (has_gt_) item("truth", kTruth, false);
    item("estimate", kEstimate, false);
    item("raw match", kRawDot, true);
    if (has_gt_) {
        item("error <50 m", kGood, false);
        item("<150 m", kMid, false);
        item(">150 m", kBad, false);
    }
    return out;
}
