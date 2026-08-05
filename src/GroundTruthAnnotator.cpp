#include "GroundTruthAnnotator.hpp"
#include "MapUtils.hpp"
#include "CoordinateUtils.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <functional>
#include <algorithm>
#include <sys/stat.h>

namespace {

// kWidth/kHeight/kScale must match what the navigation pipeline assumes for
// the Google Static Maps image dimensions (effective_px = width*scale) --
// center_lat/center_lng come from the shared HaifaSamples.hpp config, so
// those already stay consistent between this tool and Main.cpp regardless.
// kTargetSpanM does NOT need to match Main.cpp's (which varies per-sample
// based on the altitude estimate anyway) -- each tool computes its own
// self-consistent zoom/mpp and the resulting lat/lng is span-independent.
// Widened from 1500 to 3000m: Sample 1's flight extends past the map edge
// before the flight ends. Still ~2.3 m/px at this zoom -- plenty precise
// for clicking.
constexpr int    kWidth      = 640;
constexpr int    kHeight     = 640;
constexpr int    kScale      = 2;
constexpr double kTargetSpanM = 3000.0;

struct AnnotatorState {
    cv::Mat video_frame;       // current video frame (raw BGR)
    cv::Mat map_base;          // satellite map AS DISPLAYED (rotated 90 CCW -- see
                                // map_orig_width) with markers drawn so far
    cv::Mat map_display;

    // Width of the satellite map BEFORE the 90 CCW display rotation. Needed to
    // convert a click on the rotated display back to the original map's pixel
    // space before it can be passed to CoordinateUtils::pixelToLatLng (which
    // only knows about the original, unrotated coordinate system).
    int     map_orig_width = 0;

    int     video_frame_idx = 0;
    double  fps             = 25.0;

    double  center_lat, center_lng;
    int     center_x, center_y;
    double  mpp, m_per_deg_lat, m_per_deg_lng;

    int     point_count = 0;
    std::ofstream* csv  = nullptr;
};

// Inverse of cv::rotate(..., cv::ROTATE_90_COUNTERCLOCKWISE): given a pixel
// clicked on the rotated display, returns the corresponding pixel in the
// original (unrotated) map. Verified empirically: original (x,y) -> rotated
// (y, W-1-x), so the inverse is x_orig = W-1-y_rot, y_orig = x_rot, where W
// is the ORIGINAL map's width (map_orig_width, not the rotated one).
cv::Point unrotate90CounterClockwise(int x_rot, int y_rot, int orig_width) {
    return cv::Point(orig_width - 1 - y_rot, x_rot);
}

// The crosshair marks the exact centre of the frame. Since the footage is
// nadir (straight down -- verified on all 3 samples), the centre pixel is
// the point directly beneath the drone: the same quantity the navigation
// pipeline's Raw_Lat/Filtered_Lat are trying to estimate. Ground truth is
// only meaningful if it's anchored to that same point, not to an arbitrary
// landmark elsewhere in the frame (which would be offset from the drone's
// actual position by an amount we can't correct for without a reliable
// per-frame ground-sample-distance -- exactly the thing step 2 will fix).
void redrawVideoWindow(AnnotatorState& s) {
    cv::Mat disp = s.video_frame.clone();
    cv::Point c(disp.cols / 2, disp.rows / 2);
    cv::drawMarker(disp, c, cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 40, 2);
    cv::circle(disp, c, 6, cv::Scalar(0, 255, 255), 2);

    std::ostringstream info;
    info << "Frame " << s.video_frame_idx << "  |  points saved: " << s.point_count;
    cv::putText(disp, info.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0, 255, 255), 2);
    cv::putText(disp, "Crosshair = drone's ground position (nadir). Only annotate",
                cv::Point(10, disp.rows - 40), cv::FONT_HERSHEY_SIMPLEX,
                0.55, cv::Scalar(0, 255, 255), 1);
    cv::putText(disp, "frames where a real feature sits under/near it.",
                cv::Point(10, disp.rows - 15), cv::FONT_HERSHEY_SIMPLEX,
                0.55, cv::Scalar(0, 255, 255), 1);
    cv::imshow("GT Video Frame", disp);
}

// Single click on the map: records where the CURRENT frame's crosshair
// (i.e. the drone) actually is on the ground.
void onMapMouse(int event, int x, int y, int /*flags*/, void* userdata) {
    auto* s = static_cast<AnnotatorState*>(userdata);
    if (event != cv::EVENT_LBUTTONDOWN) return;

    // (x,y) is in the ROTATED display's coordinate space -- convert back to
    // the original map's pixel space before geo-converting, or every click
    // would resolve to the wrong location.
    cv::Point orig_px = unrotate90CounterClockwise(x, y, s->map_orig_width);
    auto geo = CoordinateUtils::pixelToLatLng(
        orig_px.x, orig_px.y, s->center_lat, s->center_lng,
        s->center_x, s->center_y, s->mpp,
        s->m_per_deg_lat, s->m_per_deg_lng);

    double time_sec = s->video_frame_idx / s->fps;
    ++s->point_count;

    (*s->csv) << s->video_frame_idx << ","
              << std::fixed << std::setprecision(2) << time_sec << ","
              << std::setprecision(8) << geo.first << "," << geo.second << ","
              << "point_" << s->point_count << "\n";
    s->csv->flush();

    cv::circle(s->map_display, cv::Point(x, y), 6, cv::Scalar(0, 0, 255), -1);
    cv::putText(s->map_display, std::to_string(s->point_count),
                cv::Point(x + 8, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 255), 2);
    cv::imshow("GT Satellite Map", s->map_display);

    std::cout << "  ✓ Point " << s->point_count << " saved: frame=" << s->video_frame_idx
              << " t=" << time_sec << "s  ->  (" << std::setprecision(6)
              << geo.first << ", " << geo.second << ")" << std::endl;
}

} // namespace

void runGroundTruthAnnotator(const HaifaSampleConfig& cfg) {
    std::cout << "\n=== GROUND TRUTH ANNOTATOR: " << cfg.sample_name << " ===" << std::endl;
    std::cout << "  The yellow crosshair in the VIDEO FRAME window marks the exact centre of" << std::endl;
    std::cout << "  the frame -- the drone's own ground position (footage is nadir/straight-down)." << std::endl;
    std::cout << "  1. Scrub the trackbar (or 'a'/'d' = ±1 frame, 'A'/'D' = ±10 frames) until a" << std::endl;
    std::cout << "     recognizable feature (pier tip, road junction, building corner, breakwater)" << std::endl;
    std::cout << "     sits under or very near the crosshair." << std::endl;
    std::cout << "  2. Left-click that same real-world point on the SATELLITE MAP window." << std::endl;
    std::cout << "  Skip frames where nothing distinct is under the crosshair (blur, open water, plain roof)." << std::endl;
    std::cout << "  8-15 well-spread points per clip is plenty -- prioritize the first and last usable" << std::endl;
    std::cout << "  frames, and both sides of any long featureless stretch." << std::endl;
    std::cout << "  ⚠️  Each run OVERWRITES this sample's ground truth -- click everything you want" << std::endl;
    std::cout << "  to keep in one sitting, previous points will not be kept." << std::endl;
    std::cout << "  Press 'q' or ESC to finish.\n" << std::endl;

    const std::string apiKey = readApiKey();
    if (apiKey.empty()) {
        std::cerr << "Failed to read API key from config.ini." << std::endl;
        return;
    }

    const int effective_px = kWidth * kScale;
    int    zoom = chooseZoomForSpan(cfg.center_lat, kTargetSpanM, effective_px);
    double mpp  = metersPerPixel(cfg.center_lat, zoom);
    const double lat_rad      = cfg.center_lat * M_PI / 180.0;
    const double m_per_deg_lat = 111320.0;
    const double m_per_deg_lng = 111320.0 * std::cos(lat_rad);

    std::ostringstream map_url;
    map_url << "https://maps.googleapis.com/maps/api/staticmap?"
            << "center=" << cfg.center_lat << "," << cfg.center_lng
            << "&zoom=" << zoom << "&size=" << kWidth << "x" << kHeight
            << "&maptype=satellite&scale=" << kScale << "&key=" << apiKey;

    // Zoom is baked into the cache filename so that changing kTargetSpanM (or
    // any other zoom-affecting constant) automatically fetches a fresh map
    // instead of silently reusing a stale cached image at the wrong scale --
    // loadOrFetchMapImage() only checks whether the file exists, not whether
    // it matches the zoom this run just computed.
    std::string map_cache_path = "Images/map_groundtruth_" + cfg.sample_name
                                 + "_z" + std::to_string(zoom) + ".png";
    cv::Mat sat_map = loadOrFetchMapImage(map_cache_path, map_url.str());
    if (sat_map.empty()) {
        std::cerr << "Failed to load ground-truth reference map for " << cfg.sample_name << std::endl;
        return;
    }

    cv::VideoCapture cap(cfg.video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << cfg.video_path << std::endl;
        return;
    }
    int    total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps           = cap.get(cv::CAP_PROP_FPS);

    mkdir("CSV Files", 0755);
    std::string csv_path = "CSV Files/ground_truth_" + cfg.sample_name + ".csv";

    // Each run starts fresh and OVERWRITES any existing ground truth for this
    // sample (rather than appending), so stray/test clicks from an earlier
    // session can never silently pollute a later one. Re-click everything you
    // want to keep in a single sitting.
    {
        std::ifstream existing(csv_path);
        if (existing.good() && existing.peek() != std::ifstream::traits_type::eof())
            std::cout << "  ⚠️  Overwriting existing ground truth at " << csv_path << std::endl;
    }
    std::ofstream csv(csv_path);   // truncates on open -- no std::ios::app
    if (!csv.is_open()) {
        std::cerr << "Failed to open " << csv_path << " for writing." << std::endl;
        return;
    }
    csv << "Frame,Time_Sec,Lat,Lng,Label\n";

    AnnotatorState state;
    // Display the map rotated 90 CCW to match the video's orientation (this
    // project's Haifa footage is rotated relative to the map's north-up
    // frame -- confirmed on Sample 2 by direct visual comparison -- making
    // it hard to correlate features without tilting your head). Only the
    // DISPLAY is rotated; all geo-conversion still happens in the original
    // map's coordinate space via unrotate90CounterClockwise() in onMapMouse().
    state.map_orig_width = sat_map.cols;
    cv::rotate(sat_map, state.map_base, cv::ROTATE_90_COUNTERCLOCKWISE);
    state.map_display    = state.map_base.clone();
    state.fps            = fps > 0 ? fps : 25.0;
    state.center_lat     = cfg.center_lat;
    state.center_lng     = cfg.center_lng;
    state.center_x       = effective_px / 2;
    state.center_y       = effective_px / 2;
    state.mpp            = mpp;
    state.m_per_deg_lat  = m_per_deg_lat;
    state.m_per_deg_lng  = m_per_deg_lng;
    state.csv            = &csv;

    cv::namedWindow("GT Video Frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("GT Video Frame", 1000, 560);
    cv::namedWindow("GT Satellite Map", cv::WINDOW_NORMAL);
    cv::resizeWindow("GT Satellite Map", 900, 900);
    cv::imshow("GT Satellite Map", state.map_display);

    cv::setMouseCallback("GT Satellite Map", onMapMouse, &state);

    auto seekAndShow = [&](int frame_idx) {
        state.video_frame_idx = frame_idx;
        cap.set(cv::CAP_PROP_POS_FRAMES, frame_idx);
        if (cap.read(state.video_frame))
            redrawVideoWindow(state);
    };

    int trackbar_pos = 0;
    std::function<void(int)> onTrackbar = seekAndShow;
    cv::createTrackbar("Frame", "GT Video Frame", &trackbar_pos,
                       std::max(0, total_frames - 1),
                       [](int pos, void* userdata) {
                           auto* fn = static_cast<std::function<void(int)>*>(userdata);
                           (*fn)(pos);
                       }, &onTrackbar);

    seekAndShow(0);

    while (true) {
        int key = cv::waitKey(30) & 0xFF;
        if (key == 'q' || key == 27) break;

        // Keyboard frame stepping — precise navigation to a sharp frame is
        // fiddly with a mouse-dragged trackbar. 'a'/'d' step one frame,
        // 'A'/'D' (shift) jump 10 frames.
        int step = 0;
        if      (key == 'd') step = 1;
        else if (key == 'a') step = -1;
        else if (key == 'D') step = 10;
        else if (key == 'A') step = -10;

        if (step != 0) {
            int new_frame = state.video_frame_idx + step;
            new_frame = std::max(0, std::min(total_frames - 1, new_frame));
            if (new_frame != state.video_frame_idx) {
                trackbar_pos = new_frame;
                cv::setTrackbarPos("Frame", "GT Video Frame", trackbar_pos);
            }
        }
    }

    cv::destroyWindow("GT Video Frame");
    cv::destroyWindow("GT Satellite Map");
    csv.close();

    // Saved separately from the cached reference map (map_groundtruth_*.png),
    // which must stay pristine since it's reloaded and reused as the pixel/
    // lat-lng reference on every future run -- burning markers into it would
    // make every subsequent session load an ever-more-cluttered image.
    if (state.point_count > 0) {
        std::string annotated_path = "Images/map_groundtruth_" + cfg.sample_name + "_annotated.png";
        cv::imwrite(annotated_path, state.map_display);
        std::cout << "  Annotated snapshot saved: " << annotated_path << std::endl;
    }

    std::cout << "\n=== Ground truth session complete: " << state.point_count
              << " point(s) saved to " << csv_path << " ===" << std::endl;
}

InitializationData readGroundTruthStart(const std::string& sample_name) {
    std::string path = "CSV Files/ground_truth_" + sample_name + ".csv";
    std::ifstream f(path);
    if (!f.is_open())
        return {false, 0, {0.0, 0.0}};

    std::string line;
    std::getline(f, line);   // header

    bool   found      = false;
    int    best_frame = 0;
    double best_lat = 0.0, best_lng = 0.0;

    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string frame_s, time_s, lat_s, lng_s;
        std::getline(ss, frame_s, ',');
        std::getline(ss, time_s,  ',');
        std::getline(ss, lat_s,   ',');
        std::getline(ss, lng_s,   ',');
        if (frame_s.empty() || lat_s.empty() || lng_s.empty())
            continue;

        int frame = std::stoi(frame_s);
        if (!found || frame < best_frame) {
            found      = true;
            best_frame = frame;
            best_lat   = std::stod(lat_s);
            best_lng   = std::stod(lng_s);
        }
    }

    if (!found)
        return {false, 0, {0.0, 0.0}};
    return {true, best_frame, {best_lat, best_lng}};
}

std::vector<std::pair<double, double>> loadGroundTruthPath(const std::string& sample_name) {
    std::string path = "CSV Files/ground_truth_" + sample_name + ".csv";
    std::ifstream f(path);
    if (!f.is_open())
        return {};

    std::string line;
    std::getline(f, line);   // header

    std::vector<std::pair<int, std::pair<double, double>>> rows;
    int max_frame = 0;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string frame_s, time_s, lat_s, lng_s;
        std::getline(ss, frame_s, ',');
        std::getline(ss, time_s,  ',');
        std::getline(ss, lat_s,   ',');
        std::getline(ss, lng_s,   ',');
        if (frame_s.empty() || lat_s.empty() || lng_s.empty())
            continue;

        int frame = std::stoi(frame_s);
        rows.push_back({frame, {std::stod(lat_s), std::stod(lng_s)}});
        max_frame = std::max(max_frame, frame);
    }

    std::vector<std::pair<double, double>> path_by_frame(max_frame, {0.0, 0.0});
    for (const auto& [frame, coords] : rows) {
        if (frame >= 1 && frame <= max_frame)
            path_by_frame[frame - 1] = coords;
    }
    return path_by_frame;
}
