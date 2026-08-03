#include "OpticalFlowTracker.hpp"
#include "CoordinateUtils.hpp"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// Feature detection parameters (tuned for harbour / low-texture scenes)
// ---------------------------------------------------------------------------
static const int    MAX_FEATURES    = 300;
static const double QUALITY_LEVEL   = 0.01;   // low threshold to get features near coastline
static const double MIN_DISTANCE    = 8.0;    // pixels
static const int    BLOCK_SIZE      = 7;
static const int    LK_WIN_SIZE     = 21;     // larger window → more robustness per point
static const int    LK_MAX_LEVEL    = 3;      // pyramid depth
static const float  LK_MAX_ERR     = 20.0f;  // max per-point LK error to keep a track
static const int    MIN_TRACKED     = 8;      // below this re-detect
static const int    REFRESH_THRESH  = 50;     // merge new features when below this

// Re-localization score threshold
static const double RELOC_SCORE_THRESHOLD = 0.22;
static const double RELOC_MAX_DIST_M      = 350.0;  // ignore reloc if > this far from prediction

// Scale self-correction EMA coefficient
static const double SCALE_EMA_SLOW = 0.03;   // very conservative — only correct clear drift

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
OpticalFlowTracker::OpticalFlowTracker(
    double         gsd_init,
    const cv::Mat& satellite_map,
    double center_lat,  double center_lng,
    int    center_x,    int    center_y,
    double mpp,
    double m_per_deg_lat,
    double m_per_deg_lng,
    const cv::Mat& hud_mask)
    : gsd_(gsd_init)
    , satellite_map_(satellite_map)
    , center_lat_(center_lat), center_lng_(center_lng)
    , center_x_(center_x),     center_y_(center_y)
    , mpp_(mpp)
    , m_per_deg_lat_(m_per_deg_lat), m_per_deg_lng_(m_per_deg_lng)
    , hud_mask_(hud_mask)
    , initialized_(false)
    , lk_criteria_(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01)
    , frames_since_reloc_(0)
{
    sat_gray_  = EdgeProcessor::prepareSatelliteMap(satellite_map_);
    sat_edges_ = EdgeProcessor::computeEdgeMap(sat_gray_);
}

OpticalFlowTracker::~OpticalFlowTracker()
{
    // If a re-loc future is still running, wait for it to finish cleanly.
    if (reloc_future_.valid())
        reloc_future_.wait();
}

// ---------------------------------------------------------------------------
// detectFeatures — Shi-Tomasi corner detection with HUD mask
// ---------------------------------------------------------------------------
void OpticalFlowTracker::detectFeatures(const cv::Mat& gray)
{
    tracked_pts_.clear();
    cv::goodFeaturesToTrack(gray, tracked_pts_,
                            MAX_FEATURES, QUALITY_LEVEL, MIN_DISTANCE,
                            hud_mask_,    // mask: only detect in unmasked regions
                            BLOCK_SIZE);
}

// ---------------------------------------------------------------------------
// mergeNewFeatures — add new detections that are not near existing tracks
// ---------------------------------------------------------------------------
void OpticalFlowTracker::mergeNewFeatures(const cv::Mat& gray)
{
    std::vector<cv::Point2f> new_pts;
    cv::goodFeaturesToTrack(gray, new_pts,
                            MAX_FEATURES, QUALITY_LEVEL, MIN_DISTANCE,
                            hud_mask_, BLOCK_SIZE);

    const float min_dist_sq = (MIN_DISTANCE * 2) * (MIN_DISTANCE * 2);

    for (const auto& np : new_pts) {
        bool too_close = false;
        for (const auto& ep : tracked_pts_) {
            float dx = np.x - ep.x, dy = np.y - ep.y;
            if (dx*dx + dy*dy < min_dist_sq) { too_close = true; break; }
        }
        if (!too_close) tracked_pts_.push_back(np);
        if (static_cast<int>(tracked_pts_.size()) >= MAX_FEATURES) break;
    }
}

// ---------------------------------------------------------------------------
// estimatePosition — main per-frame call
// ---------------------------------------------------------------------------
PositionEstimate OpticalFlowTracker::estimatePosition(
    const cv::Mat&                    drone_view,
    const std::vector<ReferenceCrop>& /*reference_crops*/,
    const std::pair<double, double>&  last_position)
{
    // drone_view is expected to already be preprocessed greyscale
    // (CLAHE, HUD mask applied) by the caller (VideoProcessing.cpp).
    if (drone_view.empty())
        return PositionEstimate(last_position, 0.0, -1);

    // =========================================================================
    // INITIALISATION — first call
    // =========================================================================
    if (!initialized_ || prev_gray_.empty()) {
        current_pos_ = last_position;
        prev_gray_   = drone_view.clone();
        detectFeatures(drone_view);
        initialized_ = true;
        std::cout << "   [OptFlow] Initialized: " << tracked_pts_.size()
                  << " features, pos=(" << current_pos_.first
                  << ", " << current_pos_.second << ")" << std::endl;
        return PositionEstimate(current_pos_, 0.5, -1);
    }

    // =========================================================================
    // LUCAS-KANADE OPTICAL FLOW
    // =========================================================================
    if (tracked_pts_.empty()) {
        detectFeatures(drone_view);
        prev_gray_ = drone_view.clone();
        return PositionEstimate(current_pos_, 0.3, -1);
    }

    std::vector<cv::Point2f> next_pts;
    std::vector<uchar>       status;
    std::vector<float>       err;

    cv::calcOpticalFlowPyrLK(
        prev_gray_, drone_view,
        tracked_pts_, next_pts,
        status, err,
        cv::Size(LK_WIN_SIZE, LK_WIN_SIZE),
        LK_MAX_LEVEL,
        lk_criteria_);

    // Filter to good tracks only
    std::vector<cv::Point2f> good_old, good_new;
    good_old.reserve(tracked_pts_.size());
    good_new.reserve(tracked_pts_.size());

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] && err[i] < LK_MAX_ERR) {
            good_old.push_back(tracked_pts_[i]);
            good_new.push_back(next_pts[i]);
        }
    }

    // Too few tracks — re-detect and return last known position
    if (static_cast<int>(good_old.size()) < MIN_TRACKED) {
        std::cout << "   [OptFlow] Lost tracks (" << good_old.size()
                  << "), re-detecting..." << std::endl;
        detectFeatures(drone_view);
        prev_gray_ = drone_view.clone();
        tracked_pts_ = good_new.empty() ? std::vector<cv::Point2f>() : good_new;
        return PositionEstimate(current_pos_, 0.25, -1);
    }

    // =========================================================================
    // AFFINE PARTIAL TRANSFORM — gives (dx, dy, rotation, scale)
    // =========================================================================
    cv::Mat transform = cv::estimateAffinePartial2D(
        good_old, good_new,
        cv::noArray(), cv::RANSAC,
        /*reprojThreshold=*/3.0);

    double confidence = 0.0;
    if (!transform.empty()) {
        // Pixel displacement of the camera (opposite sign: if features move right,
        // drone moved right too — satellite frame perspective)
        double dx_px = transform.at<double>(0, 2);
        double dy_px = transform.at<double>(1, 2);

        // Metres displaced
        double dx_m = dx_px * gsd_;
        double dy_m = dy_px * gsd_;

        // Update geographic position
        // In image coordinates: +x = east, +y = south (positive down in image)
        double new_lat = current_pos_.first  - dy_m / m_per_deg_lat_;
        double new_lng = current_pos_.second + dx_m / m_per_deg_lng_;

        current_pos_ = {new_lat, new_lng};

        // Confidence based on inlier fraction (approximated from good tracks)
        confidence = std::min(1.0, static_cast<double>(good_old.size()) / MAX_FEATURES * 2.0);
        confidence = std::max(0.4, confidence);  // floor for valid flow

        // Slow GSD self-correction from affine scale component
        double affine_scale = std::sqrt(
            transform.at<double>(0, 0) * transform.at<double>(0, 0) +
            transform.at<double>(1, 0) * transform.at<double>(1, 0));

        // Only apply correction if the per-frame scale is reasonable
        // (>1 means drone descended, <1 means drone ascended)
        if (affine_scale > 0.9 && affine_scale < 1.1) {
            // EMA: slowly nudge GSD toward what affine says
            gsd_ *= (1.0 - SCALE_EMA_SLOW) + SCALE_EMA_SLOW * (1.0 / affine_scale);
        }
    } else {
        // Affine failed — use mean displacement as fallback
        double mean_dx = 0, mean_dy = 0;
        for (size_t i = 0; i < good_old.size(); ++i) {
            mean_dx += good_new[i].x - good_old[i].x;
            mean_dy += good_new[i].y - good_old[i].y;
        }
        mean_dx /= good_old.size();
        mean_dy /= good_old.size();

        current_pos_.first  -= (mean_dy * gsd_) / m_per_deg_lat_;
        current_pos_.second += (mean_dx * gsd_) / m_per_deg_lng_;
        confidence = 0.35;
    }

    // =========================================================================
    // REFRESH FEATURE SET
    // =========================================================================
    tracked_pts_ = good_new;   // update to new positions

    if (static_cast<int>(tracked_pts_.size()) < REFRESH_THRESH)
        mergeNewFeatures(drone_view);

    prev_gray_ = drone_view.clone();

    // =========================================================================
    // ASYNC PERIODIC RE-LOCALISATION
    // =========================================================================
    ++frames_since_reloc_;

    // Check if a previously launched reloc has finished
    if (reloc_future_.valid() &&
        reloc_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
    {
        PositionEstimate reloc = reloc_future_.get();
        if (reloc.confidence > RELOC_SCORE_THRESHOLD) {
            double dist = CoordinateUtils::calculateDistance(
                current_pos_, reloc.position, m_per_deg_lat_, m_per_deg_lng_);
            if (dist < RELOC_MAX_DIST_M) {
                std::cout << "   [OptFlow] Re-loc accepted: dist=" << std::fixed
                          << std::setprecision(1) << dist << "m  score="
                          << std::setprecision(3) << reloc.confidence << std::endl;
                // Gently nudge position (don't jump hard)
                current_pos_.first  = 0.7 * current_pos_.first  + 0.3 * reloc.position.first;
                current_pos_.second = 0.7 * current_pos_.second + 0.3 * reloc.position.second;
                confidence = std::max(confidence, reloc.confidence * 0.8);
            } else {
                std::cout << "   [OptFlow] Re-loc discarded (dist=" << std::fixed
                          << std::setprecision(1) << dist << "m, too far)" << std::endl;
            }
        }
    }

    // Launch a new re-loc if it's time and nothing is running
    if (frames_since_reloc_ >= RELOC_INTERVAL && !reloc_future_.valid()) {
        auto pos_copy  = current_pos_;
        auto gray_copy = drone_view.clone();   // safe to capture in async
        reloc_future_  = std::async(std::launch::async,
            [this, gray_copy, pos_copy]() {
                return this->attemptRelocalization(gray_copy, pos_copy);
            });
        frames_since_reloc_ = 0;
    }

    return PositionEstimate(current_pos_, confidence, -1);
}

// ---------------------------------------------------------------------------
// attemptRelocalization
//
// Extracts a local satellite window around predicted_pos, computes an edge
// map, and template-matches the current frame (resized to current GSD) to
// find a corrected position.  Runs in a background thread.
// ---------------------------------------------------------------------------
PositionEstimate OpticalFlowTracker::attemptRelocalization(
    const cv::Mat&                   frame_gray,
    const std::pair<double, double>& predicted)
{
    const double search_radius_m = 300.0;   // metres around predicted position

    // Convert predicted position to satellite pixel
    cv::Point pred_px = CoordinateUtils::latLngToPixel(
        predicted.first, predicted.second,
        center_lat_, center_lng_,
        center_x_, center_y_,
        mpp_, m_per_deg_lat_, m_per_deg_lng_);

    // Frame footprint on the satellite map in pixels
    int footprint_px = static_cast<int>(frame_gray.cols * gsd_ / mpp_);
    int search_px    = static_cast<int>(search_radius_m / mpp_);
    int margin       = footprint_px / 2 + search_px;

    cv::Rect roi(pred_px.x - margin, pred_px.y - margin,
                 2 * margin, 2 * margin);
    roi = roi & cv::Rect(0, 0, sat_gray_.cols, sat_gray_.rows);

    if (roi.width < footprint_px || roi.height < footprint_px)
        return PositionEstimate(predicted, 0.0, -1);

    cv::Mat sat_crop_edges = sat_edges_(roi).clone();

    // Resize frame to match current estimated scale
    cv::Mat frame_resized;
    cv::resize(frame_gray, frame_resized, cv::Size(footprint_px, footprint_px));
    cv::Mat frame_edges = EdgeProcessor::computeEdgeMap(frame_resized);

    if (frame_edges.cols >= sat_crop_edges.cols ||
        frame_edges.rows >= sat_crop_edges.rows)
        return PositionEstimate(predicted, 0.0, -1);

    cv::Mat result;
    cv::matchTemplate(sat_crop_edges, frame_edges, result, cv::TM_CCOEFF_NORMED);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    if (maxVal < RELOC_SCORE_THRESHOLD)
        return PositionEstimate(predicted, 0.0, -1);

    // Convert match location back to global satellite pixel
    int global_px = roi.x + maxLoc.x + frame_edges.cols / 2;
    int global_py = roi.y + maxLoc.y + frame_edges.rows / 2;

    auto geo = CoordinateUtils::pixelToLatLng(
        global_px, global_py,
        center_lat_, center_lng_,
        center_x_, center_y_,
        mpp_, m_per_deg_lat_, m_per_deg_lng_);

    return PositionEstimate({geo.first, geo.second},
                            static_cast<double>(maxVal), -1);
}
