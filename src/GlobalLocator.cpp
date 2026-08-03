#include "GlobalLocator.hpp"
#include "EdgeProcessor.hpp"
#include "CoordinateUtils.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <iomanip>
#include <future>
#include <vector>
#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

struct ScaleCandidate {
    double score;
    float  scale;       // frame footprint as a fraction of satellite map width
    double pixel_x;     // centre of match on the satellite map (pixels)
    double pixel_y;
};

// Run SIFT matching between a video frame and a satellite patch extracted at
// (centre_px, scale).  Returns RANSAC inlier count, or 0 on failure.
// This is used as a second-stage verification on the top edge-match clusters.
int siftVerify(const cv::Mat& frame_gray,
               const cv::Mat& satellite_gray,
               double centre_px, double centre_py,
               float scale,
               int num_sift_features = 500)
{
    // Extract satellite patch at the candidate position and scale
    int patch_w = static_cast<int>(satellite_gray.cols * scale);
    int patch_h = static_cast<int>(satellite_gray.rows * scale *
                                   (static_cast<float>(frame_gray.rows) /
                                    static_cast<float>(frame_gray.cols)));

    cv::Rect roi(static_cast<int>(centre_px - patch_w / 2),
                 static_cast<int>(centre_py - patch_h / 2),
                 patch_w, patch_h);
    roi = roi & cv::Rect(0, 0, satellite_gray.cols, satellite_gray.rows);

    if (roi.width < patch_w * 0.5 || roi.height < patch_h * 0.5)
        return 0;   // patch is mostly outside the map

    // Resize satellite patch to match frame dimensions so descriptors are comparable
    cv::Mat sat_patch;
    cv::resize(satellite_gray(roi), sat_patch, frame_gray.size(), 0, 0, cv::INTER_LINEAR);

    auto sift = cv::SIFT::create(num_sift_features);

    std::vector<cv::KeyPoint> kp_frame, kp_sat;
    cv::Mat desc_frame, desc_sat;
    sift->detectAndCompute(frame_gray, cv::noArray(), kp_frame, desc_frame);
    sift->detectAndCompute(sat_patch,  cv::noArray(), kp_sat,   desc_sat);

    if (desc_frame.empty() || desc_sat.empty() ||
        kp_frame.size() < 8 || kp_sat.size() < 8)
        return 0;

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn;
    try {
        matcher.knnMatch(desc_frame, desc_sat, knn, 2);
    } catch (...) { return 0; }

    std::vector<cv::Point2f> src, dst;
    for (const auto& m : knn) {
        if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance) {
            src.push_back(kp_frame[m[0].queryIdx].pt);
            dst.push_back(kp_sat[m[0].trainIdx].pt);
        }
    }
    if (static_cast<int>(src.size()) < 8) return 0;

    cv::Mat mask;
    cv::Mat H = cv::findHomography(src, dst, cv::RANSAC, 5.0, mask);
    if (H.empty()) return 0;

    return cv::countNonZero(mask);
}

// Run matchTemplate at one specific scale and return the best peak.
ScaleCandidate matchAtScale(
    const cv::Mat& sat_edges,
    const cv::Mat& frame_edges_full,  // full-resolution frame edge map
    float          scale,
    float          frame_aspect)      // frame height / frame width
{
    int tw = static_cast<int>(sat_edges.cols * scale);
    int th = static_cast<int>(sat_edges.cols * scale * frame_aspect);

    // Template must be strictly smaller than the source
    if (tw <= 0 || th <= 0 || tw >= sat_edges.cols || th >= sat_edges.rows)
        return {-1.0, scale, 0.0, 0.0};

    cv::Mat tmpl;
    cv::resize(frame_edges_full, tmpl, cv::Size(tw, th), 0, 0, cv::INTER_AREA);

    cv::Mat result;
    cv::matchTemplate(sat_edges, tmpl, result, cv::TM_CCOEFF_NORMED);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    // Centre of the template in satellite-pixel coordinates
    double cx = maxLoc.x + tw / 2.0;
    double cy = maxLoc.y + th / 2.0;

    return {maxVal, scale, cx, cy};
}

// Geographic distance in metres between two (lat, lng) pairs.
double geoDistMeters(double lat1, double lng1, double lat2, double lng2)
{
    const double m_per_deg_lat = 111320.0;
    double lat_rad = (lat1 + lat2) * 0.5 * M_PI / 180.0;
    double m_per_deg_lng = 111320.0 * std::cos(lat_rad);
    double dlat = (lat1 - lat2) * m_per_deg_lat;
    double dlng = (lng1 - lng2) * m_per_deg_lng;
    return std::sqrt(dlat * dlat + dlng * dlng);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// GlobalLocator::findStartingPosition
//
// Algorithm:
//
//   PHASE 0 — Offline setup:
//     • Preprocess satellite map (CLAHE, greyscale)
//     • Compute satellite Canny edge map (once)
//     • Compute temporal HUD mask from the first 30 video frames
//
//   PHASE 1 — Candidate generation:
//     Scan the first ~10 s of video (every frame_skip frames).
//     For each frame:
//       • Preprocess + edge-map
//       • Launch 7 std::async tasks, each trying a different scale
//       • Collect peaks with score > SCORE_THRESHOLD
//
//   PHASE 2 — Consensus clustering:
//     Convert each candidate pixel position to lat/lng.
//     Cluster by geographic proximity (200 m threshold).
//     Return centroid of the largest cluster + median scale.
// ---------------------------------------------------------------------------

// Scale factors to try.  Each value is the frame footprint as a fraction of
// the satellite map width.  Covers altitudes ~50 m → ~500 m for a typical
// drone camera with ~70° HFOV over a 1500 m map.  The denser spacing between
// 0.13 and 0.38 targets the most common low-altitude survey range.
static const float SCALES[] = {
    0.06f, 0.09f, 0.11f, 0.13f, 0.16f, 0.19f,
    0.23f, 0.27f, 0.32f, 0.38f, 0.45f, 0.52f
};
static const int   N_SCALES = static_cast<int>(sizeof(SCALES) / sizeof(SCALES[0]));

// Lower threshold so faint-but-real matches aren't discarded before SIFT
// verification can confirm or reject them.
static const double SCORE_THRESHOLD   = 0.15;   // min edge-match score to pass to SIFT
static const double CLUSTER_DIST_M    = 250.0;  // geographic cluster radius (metres)
static const int    SEARCH_SECONDS    = 12;     // search window from start of video

InitializationData GlobalLocator::findStartingPosition(
    const std::string& video_path,
    const cv::Mat&     satellite_map,
    double center_lat,  double center_lng,
    int    center_x,    int    center_y,
    double mpp,
    double m_per_deg_lat,
    double m_per_deg_lng,
    int    frame_skip,
    const cv::Mat&     feature_mask)
{
    if (satellite_map.empty()) {
        std::cerr << "GlobalLocator: satellite map is empty." << std::endl;
        return {false, 0, {0.0, 0.0}};
    }

    std::cout << "\n[GlobalLocator] Multi-scale edge-based initialization" << std::endl;
    std::cout << "   Video:      " << video_path     << std::endl;
    std::cout << "   Map size:   " << satellite_map.cols << "×" << satellite_map.rows << std::endl;
    std::cout << "   MPP:        " << mpp            << " m/px"  << std::endl;
    std::cout << "   Frame skip: " << frame_skip     << std::endl;

    // =========================================================================
    // PHASE 0 — One-time satellite map preprocessing
    // =========================================================================
    cv::Mat sat_gray  = EdgeProcessor::prepareSatelliteMap(satellite_map);
    cv::Mat sat_edges = EdgeProcessor::computeEdgeMap(sat_gray);

    // =========================================================================
    // PHASE 0b — Open video
    // =========================================================================
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "GlobalLocator: cannot open video: " << video_path << std::endl;
        return {false, 0, {0.0, 0.0}};
    }

    const double fps        = cap.get(cv::CAP_PROP_FPS);
    const int    max_frames = static_cast<int>(fps * SEARCH_SECONDS);

    // =========================================================================
    // PHASE 1 — Candidate generation (scan first SEARCH_SECONDS seconds)
    // =========================================================================
    struct GeoCandidate {
        double lat, lng;
        float  scale;
        double score;
        int    frame_idx;
    };
    std::vector<GeoCandidate> candidates;

    cv::Mat frame;
    int frame_idx = 0;
    int n_sampled = 0;

    std::cout << "[GlobalLocator] Scanning up to " << SEARCH_SECONDS
              << "s (" << max_frames << " frames, every " << frame_skip << ")..." << std::endl;

    while (cap.read(frame) && frame_idx < max_frames) {

        if (frame_idx % frame_skip != 0) { ++frame_idx; continue; }

        // Preprocess frame; suppress edge detection in logo/subtitle regions if mask provided.
        cv::Mat frame_gray  = EdgeProcessor::preprocessFrame(frame, feature_mask);
        cv::Mat frame_edges = EdgeProcessor::computeEdgeMap(frame_gray);
        if (!feature_mask.empty()) {
            cv::Mat mask_resized;
            cv::resize(feature_mask, mask_resized, frame_edges.size(), 0, 0, cv::INTER_NEAREST);
            frame_edges.setTo(0, ~mask_resized);
        }

        float aspect = static_cast<float>(frame.rows) /
                       static_cast<float>(frame.cols);

        // Launch one async task per scale — no shared state, trivially safe
        std::vector<std::future<ScaleCandidate>> futures;
        futures.reserve(N_SCALES);
        for (int s = 0; s < N_SCALES; ++s) {
            float sc = SCALES[s];
            futures.push_back(std::async(std::launch::async,
                [&sat_edges, &frame_edges, sc, aspect]() {
                    return matchAtScale(sat_edges, frame_edges, sc, aspect);
                }));
        }

        // Collect results
        for (auto& f : futures) {
            ScaleCandidate c = f.get();
            if (c.score < SCORE_THRESHOLD) continue;

            // Convert satellite pixel → lat/lng
            auto geo = CoordinateUtils::pixelToLatLng(
                static_cast<int>(c.pixel_x), static_cast<int>(c.pixel_y),
                center_lat, center_lng,
                center_x, center_y, mpp, m_per_deg_lat, m_per_deg_lng);

            candidates.push_back({geo.first, geo.second, c.scale, c.score, frame_idx});

            std::cout << "   frame " << frame_idx
                      << "  scale=" << c.scale
                      << "  score=" << std::fixed << std::setprecision(3) << c.score
                      << "  pos=(" << std::setprecision(5) << geo.first
                      << ", " << geo.second << ")" << std::endl;
        }

        ++n_sampled;
        ++frame_idx;
    }
    cap.release();

    std::cout << "[GlobalLocator] Sampled " << n_sampled << " frames, got "
              << candidates.size() << " candidates above threshold." << std::endl;

    if (candidates.empty()) {
        std::cerr << "[GlobalLocator] No candidates found — initialization failed." << std::endl;
        return {false, 0, {0.0, 0.0}};
    }

    // =========================================================================
    // PHASE 2 — Cluster candidates by geographic proximity
    // =========================================================================
    // Simple greedy clustering: iterate through candidates sorted by score
    // (best first) and assign each to the nearest existing cluster centre.
    std::sort(candidates.begin(), candidates.end(),
              [](const GeoCandidate& a, const GeoCandidate& b) {
                  return a.score > b.score;
              });

    struct Cluster {
        double sum_lat = 0, sum_lng = 0;
        double sum_scale = 0;
        int    count = 0;
        int    best_frame_idx = 0;
        double best_score = 0;

        void add(const GeoCandidate& c) {
            sum_lat   += c.lat;
            sum_lng   += c.lng;
            sum_scale += c.scale;
            ++count;
            if (c.score > best_score) {
                best_score     = c.score;
                best_frame_idx = c.frame_idx;
            }
        }
        double centroid_lat()   const { return sum_lat   / count; }
        double centroid_lng()   const { return sum_lng   / count; }
        float  centroid_scale() const { return static_cast<float>(sum_scale / count); }
    };

    std::vector<Cluster> clusters;
    for (const auto& c : candidates) {
        bool assigned = false;
        for (auto& cl : clusters) {
            if (geoDistMeters(c.lat, c.lng, cl.centroid_lat(), cl.centroid_lng())
                    < CLUSTER_DIST_M) {
                cl.add(c);
                assigned = true;
                break;
            }
        }
        if (!assigned) {
            Cluster nc;
            nc.add(c);
            clusters.push_back(nc);
        }
    }

    // Sort clusters descending by member count (ties broken by best edge score)
    std::sort(clusters.begin(), clusters.end(),
              [](const Cluster& a, const Cluster& b) {
                  return (a.count > b.count) ||
                         (a.count == b.count && a.best_score > b.best_score);
              });

    // =========================================================================
    // PHASE 3 — SIFT verification on the top clusters
    //
    // Edge template matching is a fast, coarse filter.  SIFT matching on the
    // extracted satellite patch gives a much more reliable geometric score.
    // We re-read the best frame for each cluster and run siftVerify().
    // The cluster with the highest SIFT inlier count wins.
    // =========================================================================
    const int MAX_VERIFY_CLUSTERS = std::min(3, static_cast<int>(clusters.size()));

    std::cout << "\n[GlobalLocator] SIFT verification of top "
              << MAX_VERIFY_CLUSTERS << " cluster(s)..." << std::endl;

    struct VerifiedCluster {
        int    cluster_idx;
        int    sift_inliers;
        double lat, lng;
        float  scale;
        int    frame_idx;
    };
    std::vector<VerifiedCluster> verified;

    cv::VideoCapture cap2(video_path);

    for (int ci = 0; ci < MAX_VERIFY_CLUSTERS; ++ci) {
        const Cluster& cl = clusters[ci];

        // Convert centroid lat/lng back to satellite pixel for patch extraction
        cv::Point sat_px = CoordinateUtils::latLngToPixel(
            cl.centroid_lat(), cl.centroid_lng(),
            center_lat, center_lng,
            center_x, center_y, mpp, m_per_deg_lat, m_per_deg_lng);

        // Re-read the frame that produced the best edge score for this cluster
        cap2.set(cv::CAP_PROP_POS_FRAMES, cl.best_frame_idx);
        cv::Mat verify_frame;
        if (!cap2.read(verify_frame)) continue;

        cv::Mat verify_gray = EdgeProcessor::preprocessFrame(verify_frame);

        int inliers = siftVerify(verify_gray, sat_gray,
                                 sat_px.x, sat_px.y,
                                 cl.centroid_scale());

        std::cout << "   Cluster " << ci
                  << "  pos=(" << std::fixed << std::setprecision(5)
                  << cl.centroid_lat() << ", " << cl.centroid_lng() << ")"
                  << "  edge_score=" << std::setprecision(3) << cl.best_score
                  << "  SIFT_inliers=" << inliers << std::endl;

        verified.push_back({ci, inliers,
                             cl.centroid_lat(), cl.centroid_lng(),
                             cl.centroid_scale(), cl.best_frame_idx});
    }
    cap2.release();

    // Pick the candidate with the most SIFT inliers; fall back to edge score
    // if SIFT failed entirely (all inliers == 0)
    const VerifiedCluster* winner = nullptr;
    if (!verified.empty()) {
        winner = &verified[0];
        for (const auto& v : verified)
            if (v.sift_inliers > winner->sift_inliers)
                winner = &v;
    }

    // If SIFT gave no inliers for any cluster, fall back to the largest cluster
    if (!winner || winner->sift_inliers == 0) {
        std::cout << "[GlobalLocator] SIFT verification inconclusive — "
                     "using largest edge cluster." << std::endl;
        const Cluster& fallback = clusters[0];
        double final_lat   = fallback.centroid_lat();
        double final_lng   = fallback.centroid_lng();
        float  final_scale = fallback.centroid_scale();
        std::cout << "\n[GlobalLocator] LOCK (edge only)  frame=" << fallback.best_frame_idx
                  << "  pos=(" << final_lat << ", " << final_lng << ")"
                  << "  scale=" << final_scale << std::endl;
        return {true, fallback.best_frame_idx, {final_lat, final_lng}};
    }

    std::cout << "\n[GlobalLocator] LOCK ACQUIRED (SIFT verified)" << std::endl;
    std::cout << "   SIFT inliers:  " << winner->sift_inliers << std::endl;
    std::cout << "   Frame:         " << winner->frame_idx    << std::endl;
    std::cout << "   Position:      (" << winner->lat << ", " << winner->lng << ")" << std::endl;
    std::cout << "   Scale:         " << winner->scale
              << "  GSD=" << (mpp * winner->scale) << " m/px" << std::endl;

    return {true, winner->frame_idx, {winner->lat, winner->lng}};
}
