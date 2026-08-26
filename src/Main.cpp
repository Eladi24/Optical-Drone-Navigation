#include <curl/curl.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include "FlightSimulation.hpp"
#include "CoordinateUtils.hpp"
#include "VideoProcessing.hpp"
#include "GlobalLocator.hpp"
#include "VideoPreprocessor.hpp"
#include "MapUtils.hpp"
#include "HaifaSamples.hpp"
#include "GroundTruthAnnotator.hpp"
#include "CameraModel.hpp"
#include "TelemetryImporter.hpp"
#include "DatasetVideoAssembler.hpp"
#include "DatasetSamples.hpp"
#include "SelfReferentialExperiment.hpp"
#include <fstream>


struct PathConfig
{
    std::string name;
    std::string color;
    std::string start_marker_color;
    std::string end_marker_color;
    std::vector<std::pair<double, double>> waypoints;
    std::vector<std::pair<double, double>> sample_points;
};

void processAndSimulatePath(
    const PathConfig &config,
    const cv::Mat &clean_map,
    const std::string &apiKey,
    double lat, double lng, int zoom, int width, int height,
    int scale, const std::string &maptype,
    int center_x, int center_y, double mpp, int crop_size,
    double meters_per_degree_lat, double meters_per_degree_lng,
    PositionAlgorithm algorithm)
{
    // Generate map URL
    std::ostringstream url_ss;
    url_ss << "https://maps.googleapis.com/maps/api/staticmap?"
           << "center=" << lat << "," << lng
           << "&zoom=" << zoom << "&size=" << width << "x" << height
           << "&maptype=" << maptype << "&scale=" << scale
           << "&path=color:" << config.color << "|weight:5";

    for (const auto &point : config.waypoints)
    {
        url_ss << "|" << point.first << "," << point.second;
    }

    url_ss << "&markers=color:" << config.start_marker_color << "|label:S|"
           << config.waypoints.front().first << "," << config.waypoints.front().second
           << "&markers=color:" << config.end_marker_color << "|label:E|"
           << config.waypoints.back().first << "," << config.waypoints.back().second;

    url_ss << "&markers=color:yellow|size:small";
    for (const auto &point : config.sample_points)
    {
        url_ss << "|" << point.first << "," << point.second;
    }
    url_ss << "&key=" << apiKey;

    cv::Mat path_map = loadOrFetchMapImage(
        "Images/map_" + config.name + ".png",
        url_ss.str());

    if (path_map.empty())
    {
        std::cerr << "Failed to load map for " << config.name << "\n";
        return;
    }

    // Generate crops
    std::vector<ReferenceCrop> crops;
    for (const auto &point : config.sample_points)
    {
        // Convert lat/lng to pixel coordinates on the map
        cv::Point pt = CoordinateUtils::latLngToPixel(
            point.first, point.second,
            lat, lng, center_x, center_y, mpp,
            meters_per_degree_lat, meters_per_degree_lng);
        
        // Define crop rectangle centered on the point
        cv::Rect crop_rect(pt.x - crop_size / 2, pt.y - crop_size / 2,
                           crop_size, crop_size);
        crop_rect = crop_rect & cv::Rect(0, 0, clean_map.cols, clean_map.rows);
        
        // Only add valid crops that have a positive area
        if (crop_rect.width > 0 && crop_rect.height > 0)
        {
            cv::Mat cropped = clean_map(crop_rect).clone();
            crops.push_back({point, cropped});
        }
    }

    // Display
    cv::namedWindow("Map with " + config.name + " Path", cv::WINDOW_NORMAL);
    cv::imshow("Map with " + config.name + " Path", path_map);

    for (int i = 0; i < crops.size(); i++)
    {
        std::ostringstream window_name;
        window_name << config.name << " Crop " << i << " - Coordinates: "
                    << std::fixed << std::setprecision(6)
                    << crops[i].coordinates.first << ", " << crops[i].coordinates.second;
        cv::namedWindow(window_name.str(), cv::WINDOW_AUTOSIZE);
        cv::imshow(window_name.str(), crops[i].image);
    }

    std::cout << "Displaying " << crops.size() << " crops from " << config.name << " path" << std::endl;
    std::cout << "Press any key to continue" << std::endl;
    cv::waitKey(0);

    // Match and simulate
    std::cout << "\nPerforming template matching for " << config.name << " path crops..." << std::endl;
    matchCropsOnMap(clean_map, crops);

    // 🆕 Extract location and path type from config.name
    // config.name format: "diagonal_manhattan" or "zigzag_jerusalem"
    size_t underscore_pos = config.name.find('_');
    std::string path_type = config.name.substr(0, underscore_pos); // "diagonal" or "zigzag"
    std::string location = config.name.substr(underscore_pos + 1); // "manhattan" or "jerusalem"

    std::cout << "\nStarting drone flight simulation for " << config.name << "..." << std::endl;
    runDroneSimulation(clean_map, crops, config.waypoints,
                       meters_per_degree_lat, meters_per_degree_lng,
                       lat, lng, center_x, center_y, mpp, algorithm,
                       location, path_type);
}

// Shared state for the interactive start-position selector
struct MapSelectionData {
    cv::Mat  map_base;         // unmodified map (used to redraw on each click)
    cv::Mat  map_display;      // displayed copy (gets redrawn)
    cv::Point selected_pixel;
    cv::Point suggested_pixel; // GlobalLocator estimate (-1,-1 if none)
    double   mpp;
    int      box_size_px;      // pixels representing ~300 m on this map
    bool     confirmed = false;
    bool     has_suggestion = false;
};

static void onMapMouseClick(int event, int x, int y, int /*flags*/, void* userdata)
{
    MapSelectionData* d = static_cast<MapSelectionData*>(userdata);
    if (event != cv::EVENT_LBUTTONDOWN) return;

    d->selected_pixel = cv::Point(x, y);
    d->confirmed      = true;

    // Redraw from base so old boxes don't accumulate
    d->map_display = d->map_base.clone();

    // Draw suggestion (green) if it exists
    if (d->has_suggestion) {
        cv::circle(d->map_display, d->suggested_pixel, 8,  cv::Scalar(0, 220, 0), -1);
        cv::circle(d->map_display, d->suggested_pixel, 10, cv::Scalar(0, 0, 0),   2);
        cv::putText(d->map_display, "GL estimate",
                    cv::Point(d->suggested_pixel.x + 12, d->suggested_pixel.y + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 220, 0), 1);
    }

    // Draw selected box (yellow) and centre dot (cyan)
    int half = d->box_size_px / 2;
    cv::Rect box(x - half, y - half, d->box_size_px, d->box_size_px);
    cv::Rect map_rect(0, 0, d->map_display.cols, d->map_display.rows);
    cv::rectangle(d->map_display, box & map_rect, cv::Scalar(0, 255, 255), 2);
    cv::circle(d->map_display, d->selected_pixel, 5, cv::Scalar(255, 255, 0), -1);

    cv::putText(d->map_display,
                "SELECTED - press ENTER to confirm, click again to move",
                cv::Point(10, d->map_display.rows - 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

    cv::imshow("Select Start Position", d->map_display);
    std::cout << "  Selected pixel (" << x << "," << y << ") -- press ENTER to confirm or click again." << std::endl;
}

// Show the satellite map and let the user click to choose the drone's starting position.
// @param suggested  Optional GlobalLocator estimate; shown as a green dot.
//                   Pass {0,0} if unavailable.
// @return           Lat/lng of the user-selected point (or map centre on 'Q'/Escape).
std::pair<double, double> manuallySelectStart(
    const cv::Mat&               map,
    double center_lat,           double center_lng,
    int    center_x,             int    center_y,
    double mpp,
    double m_per_deg_lat,        double m_per_deg_lng,
    std::pair<double,double>     suggested = {0.0, 0.0})
{
    MapSelectionData data;
    data.map_base    = map.clone();
    data.map_display = map.clone();
    data.mpp         = mpp;
    data.box_size_px = std::max(4, static_cast<int>(300.0 / mpp));

    // Mark GlobalLocator suggestion in green
    if (suggested.first != 0.0 || suggested.second != 0.0) {
        cv::Point sp = CoordinateUtils::latLngToPixel(
            suggested.first, suggested.second,
            center_lat, center_lng, center_x, center_y,
            mpp, m_per_deg_lat, m_per_deg_lng);
        data.suggested_pixel = sp;
        data.has_suggestion  = true;

        cv::circle(data.map_display, sp, 8,  cv::Scalar(0, 220, 0), -1);
        cv::circle(data.map_display, sp, 10, cv::Scalar(0, 0, 0),   2);
        cv::putText(data.map_display, "GL estimate",
                    cv::Point(sp.x + 12, sp.y + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 220, 0), 1);
    }

    cv::putText(data.map_display,
                "Click = set start | ENTER = accept GL estimate | Q/Esc = use map centre",
                cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);

    const std::string WIN = "Select Start Position";
    cv::namedWindow(WIN, cv::WINDOW_NORMAL);
    cv::resizeWindow(WIN, 1000, 1000);
    cv::imshow(WIN, data.map_display);
    cv::setMouseCallback(WIN, onMapMouseClick, &data);

    std::cout << "\n=== START POSITION SELECTOR ===" << std::endl;
    if (data.has_suggestion)
        std::cout << "  Green dot = GlobalLocator estimate ("
                  << suggested.first << ", " << suggested.second << ")" << std::endl;
    std::cout << "  LEFT-CLICK to place start, ENTER to accept GL estimate, Q to skip." << std::endl;

    // Event loop: poll until user presses Enter (confirm click), Q/Esc (skip), or clicks
    while (true) {
        int key = cv::waitKey(50) & 0xFF;

        // Enter / Return → accept: either clicked position or GL suggestion
        if (key == 13 || key == 10) {
            if (data.confirmed) break;          // clicked somewhere → use that
            if (data.has_suggestion) {          // no click, but GL exists → accept it
                data.selected_pixel = data.suggested_pixel;
                data.confirmed = true;
                break;
            }
            // Nothing to accept → use map centre
            break;
        }
        // Q or Escape → use map centre
        if (key == 'q' || key == 'Q' || key == 27) break;
    }

    cv::destroyWindow(WIN);

    if (data.confirmed) {
        auto pos = CoordinateUtils::pixelToLatLng(
            data.selected_pixel.x, data.selected_pixel.y,
            center_lat, center_lng, center_x, center_y,
            mpp, m_per_deg_lat, m_per_deg_lng);
        std::cout << "  Start position set to (" << pos.first << ", " << pos.second << ")" << std::endl;
        return pos;
    }

    std::cout << "  No selection — using map centre (" << center_lat << ", " << center_lng << ")" << std::endl;
    return {center_lat, center_lng};
}

// Shared by any pipeline that already has a geo-referenced reference map and a
// GSD-normalized video to match against it: builds the reference crop grid,
// resolves a starting position (ground truth if available, else GlobalLocator +
// manual selection), and runs processVideoNavigation. Used by both the Haifa
// pipeline (map fetched live from Google Maps) and the telemetry-labeled
// dataset pipeline (map pre-converted from the dataset's own GeoTIFF) below,
// which differ only in how ref_map/center_lat/center_lng/mpp are obtained --
// that acquisition step stays in each pipeline's own thin wrapper function.
void runReferenceMatchingPipeline(
    const std::string& video_path,
    const std::string& location_name,
    const std::string& sample_name,
    const cv::Mat& ref_map,
    double center_lat, double center_lng,
    int center_x, int center_y,
    double mpp, double m_per_deg_lat, double m_per_deg_lng,
    double video_frame_gsd,
    PositionAlgorithm algorithm,
    const cv::Mat& feature_mask,
    int frame_skip = 3,
    const std::vector<double>* frame_times_sec = nullptr,
    int grid_spacing_meters = 50,
    int start_frame_idx_override = -1,
    bool use_particle_filter = false,
    const std::vector<double>* frame_headings_deg = nullptr)
{
    // ------------------------------------------------------------------
    // Reference crop grid for ORB/SIFT/Hybrid
    // ------------------------------------------------------------------
    std::cout << "Generating reference crop grid (" << grid_spacing_meters << " m spacing, "
              << kReferenceCropMeters << " m crops)..." << std::endl;
    std::vector<ReferenceCrop> crops = generateReferenceCropsGrid(
        ref_map, center_lat, center_lng, center_x, center_y,
        mpp, m_per_deg_lat, m_per_deg_lng, grid_spacing_meters, kReferenceCropMeters);

    // ------------------------------------------------------------------
    // Starting position.
    //
    // If a ground-truth CSV already exists for this sample, its
    // earliest-frame point is a verified position -- use it directly
    // and skip GlobalLocator + manual selection entirely. Re-deriving
    // the start automatically (GlobalLocator, often inaccurate here)
    // or manually (imprecise, different every run) would only add
    // init-position noise to the very thing under test: per-frame
    // matching accuracy. Falls back to the normal flow otherwise.
    // ------------------------------------------------------------------
    InitializationData init = readGroundTruthStart(sample_name);
    if (init.success) {
        std::cout << "Using ground-truth start position (frame "
                  << init.frame_index << "): ("
                  << init.coordinates.first << ", "
                  << init.coordinates.second
                  << ") -- skipping GlobalLocator and manual selection."
                  << std::endl;
    } else {
        std::cout << "No ground truth found for " << sample_name
                  << " -- running GlobalLocator (multi-scale edge matching)..."
                  << std::endl;
        init = GlobalLocator::findStartingPosition(
            video_path, ref_map,
            center_lat, center_lng, center_x, center_y,
            mpp, m_per_deg_lat, m_per_deg_lng,
            /*frame_skip=*/10, feature_mask);

        if (!init.success) {
            std::cerr << "GlobalLocator failed for " << sample_name << std::endl;
            init.coordinates = {0.0, 0.0};  // sentinel — no suggestion
            init.frame_index = 0;
        } else {
            std::cout << "GlobalLocator estimate: ("
                      << init.coordinates.first << ", "
                      << init.coordinates.second << ") at frame "
                      << init.frame_index << std::endl;
        }

        // Manual start-position selector: shows satellite map + the
        // GlobalLocator estimate (green dot if found). User can click
        // to override or press ENTER to accept the suggestion.
        std::pair<double,double> manual_pos = manuallySelectStart(
            ref_map,
            center_lat, center_lng,
            center_x, center_y,
            mpp, m_per_deg_lat, m_per_deg_lng,
            init.coordinates);   // {0,0} if GL failed → no suggestion shown

        // If user made a valid selection, use it; otherwise fall back to GL or centre
        if (manual_pos.first != 0.0 || manual_pos.second != 0.0)
            init.coordinates = manual_pos;
        else if (!init.success)
            init.coordinates = {center_lat, center_lng};
    }

    // ------------------------------------------------------------------
    // Navigation with selected algorithm
    //
    // start_frame_idx_override lets a caller begin processing partway
    // through the video while still seeding the Kalman filter with
    // ground truth's earliest-frame position (init.coordinates) -- that
    // position is real GPS data, accurate regardless of which frame
    // processing actually starts at. Used for diagnostic batch runs that
    // want to sample frames spread across a flight without being anchored
    // to frame 0.
    // ------------------------------------------------------------------
    int effective_start_frame_idx = start_frame_idx_override >= 0
        ? start_frame_idx_override : init.frame_index;

    // Loaded here (not passed in) so every caller -- Haifa samples with
    // ground truth and the dataset pipeline alike -- gets the live
    // actual-vs-estimated overlay for free, keyed off the same sample_name
    // already used for readGroundTruthStart() above. Empty vector (not a
    // null check on the caller's part) for samples without ground truth
    // yet, e.g. Haifa Sample 3 -- processVideoNavigation only draws the
    // overlay when the vector it's given is non-null, so pass nullptr in
    // that case rather than a dangling reference to an empty local.
    std::vector<std::pair<double, double>> gt_path = loadGroundTruthPath(sample_name);

    processVideoNavigation(
        video_path, ref_map, crops,
        center_lat, center_lng, center_x, center_y,
        mpp, m_per_deg_lat, m_per_deg_lng,
        algorithm,
        location_name, sample_name,
        video_frame_gsd,
        frame_skip,
        init.coordinates, effective_start_frame_idx,
        /*x_offset=*/0, /*y_offset=*/0,
        feature_mask,
        frame_times_sec,
        gt_path.empty() ? nullptr : &gt_path,
        use_particle_filter,
        /*max_frames_to_process=*/0,
        frame_headings_deg);
}

// Runs the full video-navigation pipeline for one Haifa sample: preprocessing
// (mask, altitude, camera angle) -> altitude-derived GSD -> live satellite map
// fetch from Google Maps -> runReferenceMatchingPipeline() above.
//
// Templated on the config type (rather than hardcoded to HaifaSampleConfig) so
// another Google-Maps-backed sample source could reuse this body, as long as it
// exposes the same fields consumed below: video_path, sample_name,
// location_name, center_lat, center_lng. A dataset that brings its OWN
// pre-referenced map (no live fetch) is NOT a fit for this function -- see
// runDatasetPipelineForSample() below instead.
template <typename SampleConfig>
void runVideoPipelineForSample(const SampleConfig& cfg,
                                const std::string& apiKey,
                                PositionAlgorithm algorithm)
{
    const int    width_haifa      = 640;
    const int    height_haifa     = 640;
    const int    scale_haifa      = 2;
    const int    effective_px     = width_haifa * scale_haifa;
    const double target_span_m    = 1500.0;

    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "Processing: " << cfg.video_path << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // ------------------------------------------------------------------
    // STEP 1: Video preprocessing — mask, altitude, camera angle
    // ------------------------------------------------------------------
    std::string preproc_out = "Videos/preprocessed_" + cfg.sample_name + ".avi";
    VideoPreprocessor preprocessor(cfg.video_path, preproc_out, "masks");
    VideoMetadata meta = preprocessor.process(/*frame_skip=*/3);

    if (meta.total_frames == 0) {
        std::cerr << "Could not open video, skipping: " << cfg.video_path << std::endl;
        return;
    }

    cv::Mat feature_mask = meta.feature_mask;

    // ------------------------------------------------------------------
    // STEP 2: Altitude-derived scale.
    //
    // video_frame_gsd (metres/pixel of the RAW video frame) is needed
    // so processVideoNavigation can resize its per-frame crop to the
    // same ground-sample-distance as the reference crops before
    // ORB/SIFT/Hybrid matching -- without this the two images being
    // matched can represent wildly different real-world footprints.
    //
    // If the interactive altitude step was skipped, fall back to a
    // typical broadcast/survey-drone altitude so matching still has a
    // sane scale to work with (degraded accuracy, not a crash) --
    // and warn, since a real estimate is what fixes this properly.
    // ------------------------------------------------------------------
    constexpr double kFallbackAltitudeM  = 120.0;

    double altitude_m = meta.estimated_mean_altitude;
    if (altitude_m <= 0.0) {
        altitude_m = kFallbackAltitudeM;
        std::cerr << "  No altitude estimate for " << cfg.sample_name
                  << " -- falling back to " << kFallbackAltitudeM
                  << "m. Re-run the preprocessor and complete the "
                  << "altitude step for a real estimate." << std::endl;
    }
    double frame_footprint_m = CameraModel::frameFootprintMeters(altitude_m);
    double video_frame_gsd   = CameraModel::groundSampleDistance(altitude_m, meta.width);

    // Map fetching: zoom level chosen from the (real, not fallback)
    // altitude estimate if available; falls back to 1500m span target.
    double span_m = target_span_m;
    if (meta.estimated_mean_altitude > 0) {
        // Fetch a map that is ~3x the estimated drone footprint width
        // so GlobalLocator has room to search
        span_m = frame_footprint_m * 3.0;
        std::cout << "Altitude estimate: " << meta.estimated_mean_altitude
                  << " m  -> map span: " << span_m << " m" << std::endl;
    }

    int zoom = chooseZoomForSpan(cfg.center_lat, span_m, effective_px);
    double mpp = metersPerPixel(cfg.center_lat, zoom);
    const double lat_rad = cfg.center_lat * M_PI / 180.0;
    const double m_per_deg_lat = 111320.0;
    const double m_per_deg_lng = 111320.0 * std::cos(lat_rad);

    // Build the Google Maps API URL for this area
    std::ostringstream map_url;
    map_url << "https://maps.googleapis.com/maps/api/staticmap?"
            << "center=" << cfg.center_lat << "," << cfg.center_lng
            << "&zoom=" << zoom
            << "&size=" << width_haifa << "x" << height_haifa
            << "&maptype=satellite&scale=" << scale_haifa
            << "&key=" << apiKey;

    // Include zoom in filename so a new altitude estimate triggers a fresh fetch
    std::string map_cache_path = "Images/map_clean_" + cfg.location_name
                                 + "_z" + std::to_string(zoom) + ".png";
    cv::Mat ref_map = loadOrFetchMapImage(map_cache_path, map_url.str());
    if (ref_map.empty()) {
        std::cerr << "Failed to load map for " << cfg.sample_name << ", skipping." << std::endl;
        return;
    }

    runReferenceMatchingPipeline(
        cfg.video_path, cfg.location_name, cfg.sample_name,
        ref_map, cfg.center_lat, cfg.center_lng, effective_px/2, effective_px/2,
        mpp, m_per_deg_lat, m_per_deg_lng,
        video_frame_gsd, algorithm, feature_mask, /*frame_skip=*/3);
}

// Runs the full pipeline for one telemetry-labeled dataset sample (see
// DatasetSamples.hpp). Differs from runVideoPipelineForSample() (Haifa) in
// exactly the two places that genuinely differ for this data source:
//   - the video doesn't exist yet -- it's assembled once from the dataset's
//     ordered drone images (DatasetVideoAssembler)
//   - the reference map is pre-converted from the dataset's own GeoTIFF and
//     geo-referenced from its own bounding-box CSV, not fetched live from
//     Google Maps
// Both real per-frame altitude (skips VideoPreprocessor's interactive
// line-draw entirely, via the same altitude-cache file it already checks)
// and real per-frame capture times (fed to processVideoNavigation's
// frame_times_sec, since this dataset's capture intervals are NOT constant --
// survey-flight turnarounds take far longer between frames) come from the
// dataset's own telemetry instead of being estimated or assumed. Everything
// after that -- reference crop grid, starting position, matching -- is the
// same shared runReferenceMatchingPipeline() Haifa uses.
void runDatasetPipelineForSample(const DatasetSampleConfig& cfg,
                                  PositionAlgorithm algorithm,
                                  bool use_heading_prewarp = false)
{
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "Processing dataset sample: " << cfg.sample_name << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    mkdir("Videos",    0755);
    mkdir("CSV Files", 0755);
    mkdir("masks",     0755);

    // ------------------------------------------------------------------
    // STEP 1: Import telemetry -> ground_truth_<sample>.csv (real GPS,
    // replacing manual annotation entirely) + telemetry_<sample>.csv
    // (dense per-frame altitude/heading).
    // ------------------------------------------------------------------
    TelemetryImportResult telemetry = importUavVisLocTelemetry(cfg.telemetry_csv, cfg.sample_name);
    if (!telemetry.success) {
        std::cerr << "Failed to import telemetry for " << cfg.sample_name << ", skipping." << std::endl;
        return;
    }

    // ------------------------------------------------------------------
    // STEP 2: Assemble the ordered images into a synthetic video, so
    // VideoPreprocessor and processVideoNavigation's cv::VideoCapture-based
    // frame loop work unchanged (see DatasetVideoAssembler.hpp -- this
    // video's own fps is nominal only, real timing comes from telemetry).
    // ------------------------------------------------------------------
    if (!assembleDatasetVideo(cfg.image_dir, cfg.image_prefix, cfg.frame_count, cfg.video_path)) {
        std::cerr << "Failed to assemble video for " << cfg.sample_name << ", skipping." << std::endl;
        return;
    }

    // ------------------------------------------------------------------
    // STEP 3: Pre-seed VideoPreprocessor's altitude cache with the real
    // telemetry altitude, so its interactive line-draw step is skipped
    // entirely (same cache file/mechanism it already uses for Haifa's
    // manual estimate -- see VideoPreprocessor.cpp). Only written if not
    // already cached, so a manual override (if ever needed) isn't clobbered.
    // ------------------------------------------------------------------
    std::string stem = cfg.video_path;
    for (char& c : stem) if (c == '/' || c == '\\' || c == ' ') c = '_';
    std::string altitude_cache_path = "masks/" + stem + "_altitude.txt";
    {
        std::ifstream existing(altitude_cache_path);
        if (!existing.good()) {
            std::ofstream out(altitude_cache_path);
            out << telemetry.mean_altitude_m;
        }
    }

    // Same idea for VideoPreprocessor's mask cache: these are clean aerial
    // survey photos (no HUD/logo/subtitle overlays like Haifa's broadcast
    // footage), so "exclude nothing" (all 255) is the actually-correct mask
    // here, not just an automation shortcut -- pre-seeding it skips the
    // interactive mask editor the same way the altitude cache above skips
    // the interactive altitude line-draw.
    std::string mask_cache_path = "masks/" + stem + "_mask.png";
    {
        std::ifstream existing(mask_cache_path);
        if (!existing.good()) {
            cv::VideoCapture probe(cfg.video_path);
            cv::Mat first_frame;
            if (probe.isOpened() && probe.read(first_frame)) {
                cv::Mat blank_mask(first_frame.size(), CV_8UC1, cv::Scalar(255));
                cv::imwrite(mask_cache_path, blank_mask);
            }
        }
    }

    std::string preproc_out = "Videos/preprocessed_" + cfg.sample_name + ".avi";
    VideoPreprocessor preprocessor(cfg.video_path, preproc_out, "masks");
    VideoMetadata meta = preprocessor.process(/*frame_skip=*/1);

    if (meta.total_frames == 0) {
        std::cerr << "Could not open assembled video, skipping: " << cfg.video_path << std::endl;
        return;
    }

    cv::Mat feature_mask = meta.feature_mask;

    double altitude_m = meta.estimated_mean_altitude > 0 ? meta.estimated_mean_altitude
                                                          : telemetry.mean_altitude_m;
    double video_frame_gsd = CameraModel::groundSampleDistance(altitude_m, meta.width);

    // ------------------------------------------------------------------
    // STEP 4: Load the pre-converted local satellite map and geo-reference
    // it from the dataset's own bounding-box CSV (see
    // scripts/convert_uavvisloc_satellite.py) -- no Google Maps fetch.
    // ------------------------------------------------------------------
    SatelliteBounds bounds = readUavVisLocSatelliteBounds(cfg.coords_csv, cfg.map_name);
    if (!bounds.success) {
        std::cerr << "No bounding box for " << cfg.map_name << " in " << cfg.coords_csv
                  << ", skipping." << std::endl;
        return;
    }

    cv::Mat ref_map = cv::imread(cfg.satellite_png);
    if (ref_map.empty()) {
        std::cerr << "Failed to load " << cfg.satellite_png
                  << " -- run scripts/convert_uavvisloc_satellite.py first, skipping." << std::endl;
        return;
    }

    double center_lat = (bounds.lt_lat + bounds.rb_lat) / 2.0;
    double center_lng = (bounds.lt_lon + bounds.rb_lon) / 2.0;
    const double lat_rad = center_lat * M_PI / 180.0;
    const double m_per_deg_lat = 111320.0;
    const double m_per_deg_lng = 111320.0 * std::cos(lat_rad);
    double width_m  = (bounds.rb_lon - bounds.lt_lon) * m_per_deg_lng;
    double height_m = (bounds.lt_lat - bounds.rb_lat) * m_per_deg_lat;
    double mpp = ((width_m / ref_map.cols) + (height_m / ref_map.rows)) / 2.0;

    // ------------------------------------------------------------------
    // STEP 5: Real per-frame capture times -> Kalman filter's dt, instead
    // of the constant frame_skip/fps every other pipeline here uses.
    // ------------------------------------------------------------------
    std::vector<double> frame_times_sec = loadTelemetryFrameTimes(cfg.sample_name);

    // STRATEGY.md Phase 2 telemetry-as-prior pre-warp: opt-in (default off,
    // so every existing benchmark stays byte-for-byte reproducible) per-frame
    // heading, used by processVideoNavigation to north-align the drone crop
    // before matching. See CLAUDE.md's Phase 2 Investigation Log for the
    // empirical sign verification.
    std::vector<double> frame_headings_deg;
    if (use_heading_prewarp)
        frame_headings_deg = loadTelemetryHeadings(cfg.sample_name);

    // UAV-VisLoc's map covers a much larger real-world area (~7-9km) than
    // Haifa's (~1.5km), so the default 50m grid spacing would generate tens
    // of thousands of reference crops -- 200m keeps the crop count in the
    // same rough ballpark as Haifa's while still giving ORB/SIFT/Hybrid a
    // dense enough search grid.
    runReferenceMatchingPipeline(
        cfg.video_path, cfg.location_name, cfg.sample_name,
        ref_map, center_lat, center_lng, ref_map.cols / 2, ref_map.rows / 2,
        mpp, m_per_deg_lat, m_per_deg_lng,
        video_frame_gsd, algorithm, feature_mask,
        /*frame_skip=*/1, &frame_times_sec, /*grid_spacing_meters=*/200,
        /*start_frame_idx_override=*/-1,
        // Restored to the Kalman baseline (was TEMPly true for A/B testing
        // ParticleFilter vs DroneKalmanFilter -- see CLAUDE.md Investigation
        // Log's particle-filter section, paused not abandoned). STRATEGY.md
        // Phase 0's gate measures the fixed single-hypothesis Kalman/RANSAC
        // path specifically; leaving this on the particle filter would
        // measure a different, already-characterized code path instead.
        /*use_particle_filter=*/false,
        frame_headings_deg.empty() ? nullptr : &frame_headings_deg);
}

int main(int argc, char **argv)
{
    PositionAlgorithm algorithm = PositionAlgorithm::HYBRID;
    bool run_jerusalem = true;
    bool run_manhattan = true;
    bool run_haifa = true;
    // Telemetry-labeled dataset validation pipeline (see DatasetSamples.hpp) --
    // opt-in only via an explicit "d" 2nd arg, never part of the j+m+h default,
    // since it needs its own Datasets/ files and doesn't share a Google Maps key.
    bool run_dataset = false;
    // Same-domain self-referential diagnostic (see SelfReferentialExperiment.hpp
    // and CLAUDE.md Investigation Log) -- opt-in via "s", same reasoning as "d".
    bool run_selfref = false;
    if (argc > 1)
    {
        std::string algo_str = argv[1];
        if (algo_str == "orb" || algo_str == "ORB")
            algorithm = PositionAlgorithm::ORB;
        else if (algo_str == "sift" || algo_str == "SIFT")
            algorithm = PositionAlgorithm::SIFT;
        else if (algo_str == "hybrid" || algo_str == "HYBRID")
            algorithm = PositionAlgorithm::HYBRID;
        else if (algo_str == "optical_flow" || algo_str == "OPTICAL_FLOW")
            algorithm = PositionAlgorithm::OPTICAL_FLOW;
        else if (algo_str == "akaze" || algo_str == "AKAZE")
            algorithm = PositionAlgorithm::AKAZE;
        else if (algo_str == "split" || algo_str == "SPLIT")
            algorithm = PositionAlgorithm::SPLIT;
        else if (algo_str == "split_deit" || algo_str == "SPLIT_DEIT")
            algorithm = PositionAlgorithm::SPLIT_LEARNED_RETRIEVAL;
        else if (algo_str == "split_xfeat" || algo_str == "SPLIT_XFEAT")
            algorithm = PositionAlgorithm::SPLIT_LEARNED_MATCH_XFEAT;
        else if (algo_str == "split_learned" || algo_str == "SPLIT_LEARNED")
            algorithm = PositionAlgorithm::SPLIT_LEARNED;
        else if (algo_str == "split_disk" || algo_str == "SPLIT_DISK")
            algorithm = PositionAlgorithm::SPLIT_LEARNED_MATCH_DISK_LIGHTGLUE;
        else if (algo_str == "split_learned_disk" || algo_str == "SPLIT_LEARNED_DISK")
            algorithm = PositionAlgorithm::SPLIT_LEARNED_DISK;
        else
            std::cout << "Unknown algorithm '" << algo_str << "', using HYBRID." << std::endl;
    }
    if (argc > 2)
    {
        std::string sim_str = argv[2];
        run_jerusalem = false, run_manhattan = false, run_haifa = false;
        if (sim_str == "j")
            run_jerusalem = true;
        else if (sim_str == "m")
            run_manhattan = true;
        else if (sim_str == "h")
            run_haifa = true;
        else if (sim_str == "d")
            run_dataset = true;
        else if (sim_str == "s")
            run_selfref = true;
        else
            run_jerusalem = run_manhattan = run_haifa = true;
    }
    // Optional 4th arg: a per-sample filter, meaning differs by location --
    // restricts the Haifa loop to a single sample number (1-3) when
    // location=h (existing behavior), or does a substring match against
    // DatasetSampleConfig::sample_name when location=d (e.g. "uavvisloc_01"
    // or just "01") -- lets scripts/run_benchmark.py invoke one flight at a
    // time instead of always running every configured flight.
    std::string sample_filter_arg;
    if (argc > 3)
        sample_filter_arg = argv[3];
    int haifa_sample_filter = sample_filter_arg.empty() ? 0 : std::atoi(sample_filter_arg.c_str());

    // Only Jerusalem/Manhattan/Haifa fetch live Google Maps imagery -- the
    // dataset pipeline brings its own pre-converted reference map, so a
    // dataset-only run shouldn't require config.ini to exist at all.
    std::string apiKey;
    if (run_jerusalem || run_manhattan || run_haifa)
    {
        apiKey = readApiKey();
        if (apiKey.empty())
        {
            std::cerr << "Failed to read API key. Please check your config.ini file." << std::endl;
            return 1;
        }
    }

    // Google Static Maps API max size per image is 640x640 at scale=2, which gives us an effective 1280x1280 map to work with
    const int width = 640; 
    const int height = 640;
    const int scale = 2;
    const std::string maptype = "satellite";

    // We want to cover a span of approximately 1000 meters across the width of the image
    const double target_span_m = 1000.0;
    const int effective_px = width * scale;
    const double path_length_m = 350.0;
    const double diagonal_component_m = path_length_m / sqrt(2.0);

    // ==================== JERUSALEM SIMULATION ====================
    if (run_jerusalem)
    {
        std::cout << "\n"
                  << std::string(80, '=') << std::endl;
        std::cout << "🚁 JERUSALEM SIMULATION" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        const double lat_jer = 31.7767;
        const double lng_jer = 35.2345;

        int zoom_jer = chooseZoomForSpan(lat_jer, target_span_m, effective_px);

        const double lat_radians_jer = lat_jer * M_PI / 180.0;
        const double meters_per_degree_lat_jer = 111320.0;
        const double meters_per_degree_lng_jer = 111320.0 * std::cos(lat_radians_jer);

        const double lat_offset_start_jer = -250.0 / meters_per_degree_lat_jer;
        const double lng_offset_start_jer = -250.0 / meters_per_degree_lng_jer;

        const double lat_offset_path_jer = diagonal_component_m / meters_per_degree_lat_jer;
        const double lng_offset_path_jer = diagonal_component_m / meters_per_degree_lng_jer;

        const double path_start_lat_jer = lat_jer + lat_offset_start_jer + lat_offset_path_jer;
        const double path_start_lng_jer = lng_jer + lng_offset_start_jer + lng_offset_path_jer;

        const double path_end_lat_jer = path_start_lat_jer + lat_offset_path_jer;
        const double path_end_lng_jer = path_start_lng_jer + lng_offset_path_jer;

        double mpp_jer = metersPerPixel(lat_jer, zoom_jer);
        double span_m_jer = mpp_jer * effective_px;
        std::cout << "Jerusalem - Chosen zoom=" << zoom_jer
                  << " → mpp=" << mpp_jer
                  << " → width span ≈ " << span_m_jer << " m\n";

        std::ostringstream clean_url_ss_jer;
        clean_url_ss_jer << "https://maps.googleapis.com/maps/api/staticmap?"
                         << "center=" << lat_jer << "," << lng_jer
                         << "&zoom=" << zoom_jer
                         << "&size=" << width << "x" << height
                         << "&maptype=" << maptype
                         << "&scale=" << scale
                         << "&key=" << apiKey;

        cv::Mat clean_map_jer = loadOrFetchMapImage(
            "Images/map_clean_jerusalem.png",
            clean_url_ss_jer.str());

        if (clean_map_jer.empty())
        {
            std::cerr << "Failed to load Jerusalem map\n";
            return 1;
        }

        double pixels_per_100m_jer = 100.0 / mpp_jer;
        int crop_size_jer = static_cast<int>(std::round(pixels_per_100m_jer));
        int center_x_jer = (width * scale) / 2;
        int center_y_jer = (height * scale) / 2;

        // Diagonal Path - Jerusalem
        PathConfig diagonal_path_jer;
        diagonal_path_jer.name = "diagonal_jerusalem";
        diagonal_path_jer.color = "0x0000FF";
        diagonal_path_jer.start_marker_color = "red";
        diagonal_path_jer.end_marker_color = "green";
        diagonal_path_jer.waypoints = {{path_start_lat_jer, path_start_lng_jer}, {path_end_lat_jer, path_end_lng_jer}};

        for (int i = 0; i < 10; i++)
        {
            double t = i / 9.0;
            diagonal_path_jer.sample_points.push_back({path_start_lat_jer + t * (path_end_lat_jer - path_start_lat_jer),
                                                       path_start_lng_jer + t * (path_end_lng_jer - path_start_lng_jer)});
        }

        processAndSimulatePath(diagonal_path_jer, clean_map_jer, apiKey, lat_jer, lng_jer, zoom_jer,
                               width, height, scale, maptype, center_x_jer, center_y_jer,
                               mpp_jer, crop_size_jer, meters_per_degree_lat_jer,
                               meters_per_degree_lng_jer, algorithm);

        // Zigzag Path - Jerusalem
        PathConfig zigzag_path_jer;
        zigzag_path_jer.name = "zigzag_jerusalem";
        zigzag_path_jer.color = "0xFF0000";
        zigzag_path_jer.start_marker_color = "blue";
        zigzag_path_jer.end_marker_color = "purple";

        double step_size_jer = lat_offset_path_jer / 4.0;
        zigzag_path_jer.waypoints = {
            {path_start_lat_jer, path_start_lng_jer},
            {path_start_lat_jer + step_size_jer, path_start_lng_jer + step_size_jer},
            {path_start_lat_jer + 2 * step_size_jer, path_start_lng_jer},
            {path_start_lat_jer + 3 * step_size_jer, path_start_lng_jer + step_size_jer},
            {path_start_lat_jer + 4 * step_size_jer, path_start_lng_jer},
            {path_end_lat_jer, path_start_lng_jer + 0.5 * step_size_jer}};

        for (size_t i = 0; i < zigzag_path_jer.waypoints.size() - 1; i++)
        {
            const auto &start = zigzag_path_jer.waypoints[i];
            const auto &end = zigzag_path_jer.waypoints[i + 1];
            int points_per_segment = (i < zigzag_path_jer.waypoints.size() - 2) ? 2 : 3;

            for (int j = 0; j < points_per_segment; j++)
            {
                double t = j / static_cast<double>(points_per_segment);
                zigzag_path_jer.sample_points.push_back({start.first + t * (end.first - start.first),
                                                         start.second + t * (end.second - start.second)});
            }
        }
        zigzag_path_jer.sample_points.push_back(zigzag_path_jer.waypoints.back());

        processAndSimulatePath(zigzag_path_jer, clean_map_jer, apiKey, lat_jer, lng_jer, zoom_jer,
                               width, height, scale, maptype, center_x_jer, center_y_jer,
                               mpp_jer, crop_size_jer, meters_per_degree_lat_jer,
                               meters_per_degree_lng_jer, algorithm);
    }

    // ==================== MANHATTAN SIMULATION ====================
    if (run_manhattan)
    {
        std::cout << "\n"
                  << std::string(80, '=') << std::endl;
        std::cout << "🗽 MANHATTAN SIMULATION" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        const double lat_nyc = 40.7580; // Central Park area
        const double lng_nyc = -73.9855;

        int zoom_nyc = chooseZoomForSpan(lat_nyc, target_span_m, effective_px);

        const double lat_radians_nyc = lat_nyc * M_PI / 180.0;
        const double meters_per_degree_lat_nyc = 111320.0;
        const double meters_per_degree_lng_nyc = 111320.0 * std::cos(lat_radians_nyc);

        const double lat_offset_start_nyc = -250.0 / meters_per_degree_lat_nyc;
        const double lng_offset_start_nyc = -250.0 / meters_per_degree_lng_nyc;

        const double lat_offset_path_nyc = diagonal_component_m / meters_per_degree_lat_nyc;
        const double lng_offset_path_nyc = diagonal_component_m / meters_per_degree_lng_nyc;

        const double path_start_lat_nyc = lat_nyc + lat_offset_start_nyc + lat_offset_path_nyc;
        const double path_start_lng_nyc = lng_nyc + lng_offset_start_nyc + lng_offset_path_nyc;

        const double path_end_lat_nyc = path_start_lat_nyc + lat_offset_path_nyc;
        const double path_end_lng_nyc = path_start_lng_nyc + lng_offset_path_nyc;

        double mpp_nyc = metersPerPixel(lat_nyc, zoom_nyc);
        double span_m_nyc = mpp_nyc * effective_px;
        std::cout << "Manhattan - Chosen zoom=" << zoom_nyc
                  << " → mpp=" << mpp_nyc
                  << " → width span ≈ " << span_m_nyc << " m\n";

        std::ostringstream clean_url_ss_nyc;
        clean_url_ss_nyc << "https://maps.googleapis.com/maps/api/staticmap?"
                         << "center=" << lat_nyc << "," << lng_nyc
                         << "&zoom=" << zoom_nyc
                         << "&size=" << width << "x" << height
                         << "&maptype=" << maptype
                         << "&scale=" << scale
                         << "&key=" << apiKey;

        cv::Mat clean_map_nyc = loadOrFetchMapImage(
            "Images/map_clean_manhattan.png",
            clean_url_ss_nyc.str());

        if (clean_map_nyc.empty())
        {
            std::cerr << "Failed to load Manhattan map\n";
            return 1;
        }

        double pixels_per_100m_nyc = 100.0 / mpp_nyc;
        int crop_size_nyc = static_cast<int>(std::round(pixels_per_100m_nyc));
        int center_x_nyc = (width * scale) / 2;
        int center_y_nyc = (height * scale) / 2;

        // Diagonal Path - Manhattan
        PathConfig diagonal_path_nyc;
        diagonal_path_nyc.name = "diagonal_manhattan";
        diagonal_path_nyc.color = "0x0000FF";
        diagonal_path_nyc.start_marker_color = "red";
        diagonal_path_nyc.end_marker_color = "green";
        diagonal_path_nyc.waypoints = {{path_start_lat_nyc, path_start_lng_nyc}, {path_end_lat_nyc, path_end_lng_nyc}};

        for (int i = 0; i < 10; i++)
        {
            double t = i / 9.0;
            diagonal_path_nyc.sample_points.push_back({path_start_lat_nyc + t * (path_end_lat_nyc - path_start_lat_nyc),
                                                       path_start_lng_nyc + t * (path_end_lng_nyc - path_start_lng_nyc)});
        }

        processAndSimulatePath(diagonal_path_nyc, clean_map_nyc, apiKey, lat_nyc, lng_nyc, zoom_nyc,
                               width, height, scale, maptype, center_x_nyc, center_y_nyc,
                               mpp_nyc, crop_size_nyc, meters_per_degree_lat_nyc,
                               meters_per_degree_lng_nyc, algorithm);

        // Zigzag Path - Manhattan
        PathConfig zigzag_path_nyc;
        zigzag_path_nyc.name = "zigzag_manhattan";
        zigzag_path_nyc.color = "0xFF0000";
        zigzag_path_nyc.start_marker_color = "blue";
        zigzag_path_nyc.end_marker_color = "purple";

        double step_size_nyc = lat_offset_path_nyc / 4.0;
        zigzag_path_nyc.waypoints = {
            {path_start_lat_nyc, path_start_lng_nyc},
            {path_start_lat_nyc + step_size_nyc, path_start_lng_nyc + step_size_nyc},
            {path_start_lat_nyc + 2 * step_size_nyc, path_start_lng_nyc},
            {path_start_lat_nyc + 3 * step_size_nyc, path_start_lng_nyc + step_size_nyc},
            {path_start_lat_nyc + 4 * step_size_nyc, path_start_lng_nyc},
            {path_end_lat_nyc, path_start_lng_nyc + 0.5 * step_size_nyc}};

        for (size_t i = 0; i < zigzag_path_nyc.waypoints.size() - 1; i++)
        {
            const auto &start = zigzag_path_nyc.waypoints[i];
            const auto &end = zigzag_path_nyc.waypoints[i + 1];
            int points_per_segment = (i < zigzag_path_nyc.waypoints.size() - 2) ? 2 : 3;

            for (int j = 0; j < points_per_segment; j++)
            {
                double t = j / static_cast<double>(points_per_segment);
                zigzag_path_nyc.sample_points.push_back({start.first + t * (end.first - start.first),
                                                         start.second + t * (end.second - start.second)});
            }
        }
        zigzag_path_nyc.sample_points.push_back(zigzag_path_nyc.waypoints.back());

        processAndSimulatePath(zigzag_path_nyc, clean_map_nyc, apiKey, lat_nyc, lng_nyc, zoom_nyc,
                               width, height, scale, maptype, center_x_nyc, center_y_nyc,
                               mpp_nyc, crop_size_nyc, meters_per_degree_lat_nyc,
                               meters_per_degree_lng_nyc, algorithm);

        std::cout << "\n"
                  << std::string(80, '=') << std::endl;
        std::cout << "✅ ALL SIMULATIONS COMPLETE!" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    // ==================== HAIFA VIDEO-BASED NAVIGATION ====================
    if (run_haifa)
    {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "HAIFA PORT - VIDEO NAVIGATION PIPELINE" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        // ---- Per-sample configuration ----
        // Shared with the GroundTruth annotation tool -- see HaifaSamples.hpp.
        std::vector<HaifaSampleConfig> haifa_samples = getHaifaSamples();

        mkdir("Videos",    0755);
        mkdir("CSV Files", 0755);
        mkdir("masks",     0755);

        for (const auto& cfg : haifa_samples)
        {
            if (haifa_sample_filter > 0 &&
                cfg.sample_name != ("sample" + std::to_string(haifa_sample_filter)))
                continue;

            runVideoPipelineForSample(cfg, apiKey, algorithm);
        }

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "ALL HAIFA VIDEO MISSIONS COMPLETE!" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    // ==================== TELEMETRY-LABELED DATASET VALIDATION ====================
    if (run_dataset)
    {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "TELEMETRY-LABELED DATASET VALIDATION PIPELINE" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        std::vector<DatasetSampleConfig> dataset_samples = getDatasetSamples();
        for (const auto& cfg : dataset_samples)
        {
            if (!sample_filter_arg.empty() &&
                cfg.sample_name.find(sample_filter_arg) == std::string::npos)
                continue;

            // STRATEGY.md Phase 2 telemetry-as-prior pre-warp: opt-in via env
            // var (not a CLI flag -- this is a single-variable A/B experiment,
            // not a permanent user-facing option yet) so the default `d`
            // pipeline invocation stays byte-for-byte reproducible with every
            // already-recorded gate result. See CLAUDE.md's Phase 2
            // Investigation Log for the measured delta.
            bool use_heading_prewarp = std::getenv("DRONENAV_HEADING_PREWARP") != nullptr;
            runDatasetPipelineForSample(cfg, algorithm, use_heading_prewarp);
        }

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "ALL DATASET SAMPLES COMPLETE!" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    // ==================== SAME-DOMAIN SELF-REFERENTIAL DIAGNOSTIC ====================
    if (run_selfref)
    {
        std::vector<DatasetSampleConfig> dataset_samples = getDatasetSamples();
        for (const auto& cfg : dataset_samples)
            runSelfReferentialExperiment(cfg, algorithm);
    }

    return 0;
}
