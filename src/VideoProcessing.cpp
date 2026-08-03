#include "VideoProcessing.hpp"
#include "CoordinateUtils.hpp"
#include "Visualization.hpp"
#include "EdgeProcessor.hpp"
#include "OpticalFlowTracker.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>

std::vector<ReferenceCrop> generateReferenceCropsGrid(
    const cv::Mat &map,
    double center_lat, double center_lng,
    int center_x, int center_y,
    double mpp,
    double meters_per_degree_lat,
    double meters_per_degree_lng,
    int grid_spacing_meters,
    int crop_size_meters)
{
    std::vector<ReferenceCrop> crops;

    int crop_size_px = static_cast<int>(std::round(crop_size_meters / mpp));

    double map_width_meters  = map.cols * mpp;
    double map_height_meters = map.rows * mpp;

    // map_width (columns/x) is a LONGITUDE extent -> convert with
    // meters_per_degree_LNG; map_height (rows/y) is a LATITUDE extent ->
    // convert with meters_per_degree_LAT. These were previously swapped
    // (width divided by the lat constant, height by the lng constant) --
    // harmless near the equator where the two constants are close, but at
    // higher latitudes meters_per_degree_lng shrinks (by cos(lat)) while
    // meters_per_degree_lat stays fixed at 111320, so the swap understated
    // the longitude search range and overstated the latitude range. Found
    // via UAV-VisLoc validation (~32 deg N): understated the map's true
    // east/west extent by ~686m on each side, meaning grid points were never
    // even generated for real, in-bounds map area near those edges.
    double half_width_deg_lng  = (map_width_meters  / 2.0) / meters_per_degree_lng;
    double half_height_deg_lat = (map_height_meters / 2.0) / meters_per_degree_lat;

    double min_lat = center_lat - half_height_deg_lat;
    double max_lat = center_lat + half_height_deg_lat;
    double min_lng = center_lng - half_width_deg_lng;
    double max_lng = center_lng + half_width_deg_lng;

    double grid_spacing_lat = grid_spacing_meters / meters_per_degree_lat;
    double grid_spacing_lng = grid_spacing_meters / meters_per_degree_lng;

    std::cout << "\n📐 Generating reference crop grid:" << std::endl;
    std::cout << "   Grid spacing: " << grid_spacing_meters << "m" << std::endl;
    std::cout << "   Crop size: "    << crop_size_meters << "m (" << crop_size_px << "px)" << std::endl;
    std::cout << "   Map coverage: " << std::fixed << std::setprecision(1)
              << map_width_meters << "m × " << map_height_meters << "m" << std::endl;
    // std::fixed/setprecision are sticky on std::cout -- reset immediately so
    // this doesn't silently truncate unrelated prints later in the same run
    // (bit us once already: a later coordinate print showed (32.8, 35.0)
    // instead of the real (32.82775, 34.99480) purely from this leaking).
    std::cout << std::defaultfloat << std::setprecision(6);

    int crop_count = 0;

    for (double lat = min_lat; lat <= max_lat; lat += grid_spacing_lat) {
        for (double lng = min_lng; lng <= max_lng; lng += grid_spacing_lng) {
            cv::Point pixel = CoordinateUtils::latLngToPixel(
                lat, lng, center_lat, center_lng,
                center_x, center_y, mpp,
                meters_per_degree_lat, meters_per_degree_lng);

            cv::Rect crop_rect(
                pixel.x - crop_size_px / 2,
                pixel.y - crop_size_px / 2,
                crop_size_px, crop_size_px);
            crop_rect = crop_rect & cv::Rect(0, 0, map.cols, map.rows);

            if (crop_rect.width  > crop_size_px * 0.5 &&
                crop_rect.height > crop_size_px * 0.5)
            {
                crops.push_back(ReferenceCrop({lat, lng}, map(crop_rect).clone()));
                ++crop_count;
            }
        }
    }

    std::cout << "   ✓ Generated " << crop_count << " reference crops" << std::endl;
    return crops;
}

void processVideoNavigation(
    const std::string &video_path,
    const cv::Mat &reference_map,
    const std::vector<ReferenceCrop> &reference_crops,
    double center_lat, double center_lng,
    int center_x, int center_y,
    double mpp,
    double meters_per_degree_lat,
    double meters_per_degree_lng,
    PositionAlgorithm algorithm,
    const std::string &location_name,
    const std::string &video_name,
    double video_frame_gsd,
    int frame_skip,
    std::pair<double, double> initial_position,
    int start_frame_idx,
    int x_offset,
    int y_offset,
    const cv::Mat &feature_mask,
    const std::vector<double> *frame_times_sec)
{
    // =========================================================================
    // INITIALIZATION
    // =========================================================================
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "🎥 VIDEO-BASED NAVIGATION" << std::endl;
    std::cout << "   Location: " << location_name << "  |  Video: " << video_name << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    cv::VideoCapture video(video_path);
    if (!video.isOpened()) {
        std::cerr << "❌ Could not open: " << video_path << std::endl;
        return;
    }

    int    total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
    double fps          = video.get(cv::CAP_PROP_FPS);
    int    video_width  = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
    int    video_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));

    // std::fixed/setprecision are sticky on cout. Several later prints in this
    // function (frame progress, outlier %, avg confidence) use bare
    // setprecision(N) without their own std::fixed, implicitly assuming fixed
    // notation is already active -- without it, setprecision(N) means "N
    // significant digits" instead of "N decimal places," and any value with
    // more digits than that forces scientific notation (e.g. innovation=1125
    // prints as "1e+03"). Setting std::fixed here once, for the rest of this
    // function, is more robust than fixing each print site individually --
    // every later cout print here (percentages, confidence, innovation in
    // metres, lat/lng) is normal-magnitude and reads fine in fixed notation.
    std::cout << "\n📊 Video: " << video_width << "×" << video_height
              << "  " << total_frames << " frames  " << fps << " fps  "
              << std::fixed << std::setprecision(1) << (total_frames / fps) << "s"
              << std::setprecision(6)
              << "  (processing every " << frame_skip << " frame(s))" << std::endl;

    // =========================================================================
    // OPTICAL-FLOW BRANCH — dedicated Haifa pipeline
    // =========================================================================
    const bool use_optical_flow = (algorithm == PositionAlgorithm::OPTICAL_FLOW);

    // Build a combined feature-detection mask:
    //   margin_mask  — excludes fixed bottom/top margins (subtitles area)
    //   feature_mask — VideoPreprocessor mask for logos/static overlays
    // The intersection (AND) of both is used. Neither zeroes display pixels.
    cv::Mat margin_mask = EdgeProcessor::buildMarginMask(
        video_width, video_height,
        /*top_frac=*/0.0f, /*bottom_frac=*/0.18f,
        /*left_frac=*/0.0f, /*right_frac=*/0.0f);

    cv::Mat combined_mask;
    if (!feature_mask.empty()) {
        cv::Mat fm_resized;
        cv::resize(feature_mask, fm_resized, cv::Size(video_width, video_height),
                   0, 0, cv::INTER_NEAREST);
        cv::bitwise_and(margin_mask, fm_resized, combined_mask);
    } else {
        combined_mask = margin_mask;
    }

    cv::Mat hud_mask;
    if (use_optical_flow) {
        std::cout << "   Algorithm: Optical Flow (LK) + Kalman Filter" << std::endl;
        hud_mask = combined_mask;
    }

    // For ORB/SIFT/Hybrid: the drone-frame crop geometry (GSD-normalized to
    // match the reference crops -- see the main loop below) is fixed for the
    // whole run, since video_frame_gsd comes from a single altitude estimate.
    // Compute it once here and reuse it both for aligning the feature mask
    // (immediately below) and inside the main loop, instead of recomputing
    // per frame.
    cv::Rect legacy_crop_roi;
    int      legacy_crop_target_px = 0;
    if (!use_optical_flow) {
        // Must stay SQUARE: the reference crops are square (kReferenceCropMeters
        // x kReferenceCropMeters), and clamping width/height independently
        // against the frame's two different dimensions (1920x1080) produces a
        // non-square rectangle that then gets force-resized into a square
        // target below -- silently stretching every real-world shape and
        // corrupting the geometry ORB/SIFT rely on. Clamp both to the SAME
        // value (bounded by the smaller frame dimension) instead.
        int desired_px = static_cast<int>(std::round(kReferenceCropMeters / video_frame_gsd));
        int take = std::max(20, std::min(desired_px, std::min(video_width, video_height)));
        int sx = video_width  / 2 - take / 2 + x_offset;
        int sy = video_height / 2 - take / 2 + y_offset;
        legacy_crop_roi = cv::Rect(sx, sy, take, take) &
                          cv::Rect(0, 0, video_width, video_height);
        if (legacy_crop_roi.width > 0)
            legacy_crop_target_px = std::max(20, static_cast<int>(
                std::round(legacy_crop_roi.width * video_frame_gsd / mpp)));
    }

    // =========================================================================
    // ESTIMATOR SETUP
    // =========================================================================
    std::unique_ptr<IPositionEstimator> estimator;
    std::string algorithm_name;

    if (use_optical_flow) {
        // GSD of the raw video frame (metres/pixel), from the altitude/HFOV
        // estimate -- NOT mpp, which is the *map's* resolution and has no
        // relation to how many metres a pixel of frame-to-frame optical-flow
        // displacement represents. Using mpp here was silently baking in a
        // ~4x scale error (e.g. map mpp ~1.0 vs actual video GSD ~0.24 on
        // Sample 2) that the tracker's own EMA self-correction is deliberately
        // too slow (SCALE_EMA_SLOW=0.03) to meaningfully fix within a single
        // ~150-frame video. setGsd() exists on OpticalFlowTracker but was
        // never actually called anywhere -- this was the only place it needed
        // to be.
        double gsd_init = video_frame_gsd;

        auto* tracker = new OpticalFlowTracker(
            gsd_init, reference_map,
            center_lat, center_lng, center_x, center_y,
            mpp, meters_per_degree_lat, meters_per_degree_lng,
            hud_mask);
        estimator.reset(tracker);
        algorithm_name = "optical_flow";
    } else {
        estimator = createPositionEstimator(algorithm);
        algorithm_name = estimator->getName();

        std::cout << "   Algorithm: " << algorithm_name << " + Kalman Filter" << std::endl;
        std::cout << "   Reference crops: " << reference_crops.size() << std::endl;

        // Crop+resize the mask identically to how each drone frame is
        // cropped+resized below, so it stays spatially aligned with what's
        // actually passed to detectAndCompute(). A size-mismatched mask does
        // NOT throw in OpenCV -- it silently applies an arbitrary, misaligned
        // corner of the mask, which is worse than no mask at all.
        cv::Mat aligned_mask;
        if (legacy_crop_roi.width > 0 && legacy_crop_roi.height > 0) {
            cv::resize(combined_mask(legacy_crop_roi), aligned_mask,
                       cv::Size(legacy_crop_target_px, legacy_crop_target_px),
                       0, 0, cv::INTER_NEAREST);
        }
        estimator->setFeatureMask(aligned_mask);

        estimator->precompute(reference_crops);
    }

    // =========================================================================
    // KALMAN FILTER SETUP
    //
    // For optical flow the measurements are smooth incremental updates →
    // use lower process and measurement noise.  For ORB/SIFT the discrete
    // map-match jumps need higher measurement noise tolerance.
    // =========================================================================
    const double process_noise     = use_optical_flow ? 0.1  : 1.0;
    const double measurement_noise = use_optical_flow ? 5.0  : 20.0;

    DroneKalmanFilter kalman_filter(process_noise, measurement_noise);
    bool kalman_initialized = false;

    if (initial_position.first != 0.0 && initial_position.second != 0.0) {
        kalman_filter.initialize(initial_position.first, initial_position.second, 0.0, 0.0);
        kalman_initialized = true;
        std::cout << "   Initial position: ("
                  << initial_position.first << ", " << initial_position.second << ")" << std::endl;
    } else {
        std::cout << "   ⚠️  No initial position — will use first visual estimate." << std::endl;
    }

    // =========================================================================
    // OUTPUT SETUP
    // =========================================================================
    cv::namedWindow("Video Navigation", cv::WINDOW_NORMAL);
    cv::namedWindow("Video Frame",      cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Navigation", 1200, 800);

    mkdir("Videos",    0755);
    mkdir("CSV Files", 0755);

    std::string output_video_path = "Videos/navigation_" + algorithm_name +
                                    "_" + location_name + "_" + video_name + ".avi";
    cv::VideoWriter video_writer;
    video_writer.open(output_video_path,
                      cv::VideoWriter::fourcc('M','J','P','G'),
                      fps / frame_skip,
                      cv::Size(1200, 800), true);

    std::string telemetry_path = "CSV Files/video_telemetry_" + algorithm_name +
                                 "_" + location_name + "_" + video_name + ".csv";
    std::ofstream telemetry_file(telemetry_path);
    if (!telemetry_file.is_open()) {
        std::cerr << "❌ Could not create telemetry file." << std::endl;
        return;
    }
    telemetry_file << "Frame,Time_Sec,Raw_Lat,Raw_Lng,Predicted_Lat,Predicted_Lng,"
                   << "Filtered_Lat,Filtered_Lng,Match_Confidence,Innovation_M,"
                   << "Outlier_Rejected,Best_Match_Idx,Algorithm\n";

    // =========================================================================
    // TRAJECTORY STORAGE
    // =========================================================================
    std::vector<std::pair<double, double>> raw_positions;
    std::vector<std::pair<double, double>> filtered_positions;
    std::vector<std::pair<double, double>> predicted_positions;
    std::vector<double> confidences;
    std::vector<double> innovations_history;

    int outliers_rejected    = 0;
    int low_confidence_count = 0;

    // Lower rejection threshold for optical flow (flow updates are much smoother
    // than discrete map matches, so consecutive rejections mean genuine problems)
    const int MAX_CONSECUTIVE_REJECTIONS = use_optical_flow ? 5 : 10;
    int consecutive_rejections = 0;

    // =========================================================================
    // MAIN PROCESSING LOOP
    // =========================================================================
    std::cout << "\n🚁 Processing frames..." << std::endl;

    cv::Mat frame;
    int frame_idx        = 0;
    int processed_frames = 0;

    // Only used when frame_times_sec is supplied (see dt computation below).
    double last_real_time_sec = 0.0;
    bool   have_last_real_time = false;

    while (video.read(frame)) {

        if (frame_idx < start_frame_idx)      { ++frame_idx; continue; }
        if (frame_idx % frame_skip != 0)      { ++frame_idx; continue; }

        double time_sec, dt;
        if (frame_times_sec && frame_idx < static_cast<int>(frame_times_sec->size())) {
            time_sec = (*frame_times_sec)[frame_idx];
            dt = have_last_real_time ? (time_sec - last_real_time_sec) : (frame_skip / fps);
            last_real_time_sec = time_sec;
            have_last_real_time = true;
        } else {
            time_sec = frame_idx / fps;
            dt       = frame_skip / fps;
        }

        // --- Pre-process frame ---
        cv::Mat processed_frame;

        if (use_optical_flow) {
            // For optical flow: CLAHE + HUD mask (no ROI crop — we want full frame
            // so the tracker has the widest possible area to find features in)
            processed_frame = EdgeProcessor::preprocessFrame(frame, hud_mask);
        } else {
            // GSD-normalized crop for ORB/SIFT/Hybrid, using the geometry
            // computed once above (legacy_crop_roi/legacy_crop_target_px) so
            // it stays in lockstep with the feature mask's alignment. Resizing
            // to a footprint-matched pixel size (rather than a fixed size)
            // keeps this crop's real-world scale equal to the reference
            // crops' scale (mpp) -- without it, ORB/SIFT would be matching
            // images representing two different real-world footprints.
            if (legacy_crop_roi.width > 0 && legacy_crop_roi.height > 0) {
                cv::resize(frame(legacy_crop_roi), processed_frame,
                           cv::Size(legacy_crop_target_px, legacy_crop_target_px));
            } else {
                processed_frame = frame.clone();
            }
        }

        // =====================================================================
        // KALMAN PREDICTION
        // =====================================================================
        std::pair<double, double> predicted_position = {0.0, 0.0};
        if (kalman_initialized) {
            predicted_position = kalman_filter.predict(dt);
            predicted_positions.push_back(predicted_position);
        }

        // =====================================================================
        // VISUAL ESTIMATION
        // =====================================================================
        std::pair<double, double> last_position =
            filtered_positions.empty() ? initial_position : filtered_positions.back();
        if (last_position.first == 0.0 && last_position.second == 0.0)
            last_position = {center_lat, center_lng};

        PositionEstimate estimate;
        if (use_optical_flow) {
            // Pass the full preprocessed frame — OpticalFlowTracker handles its own
            // internal state (prev_gray, tracked_points, etc.)
            estimate = estimator->estimatePosition(
                processed_frame, reference_crops, last_position);
        } else {
            // Legacy path: full-database crop matching.
            //
            // Previously restricted to crops within 400m of the Kalman
            // filter's own predicted position. That's a feedback trap: once
            // the prediction drifts even slightly, the next search excludes
            // the true position's crops entirely, so the raw match can only
            // ever confirm the wrong neighborhood -- there is no way back.
            // The threaded Stage-1 matcher (ThreadPool/SyncBarrier in
            // ORB/SIFTFeatureEstimator) was already built to handle the full
            // crop database in parallel, so there's no need for this
            // restriction to keep per-frame latency bounded.
            estimate = estimator->estimatePosition(
                processed_frame, reference_crops, last_position);
        }

        std::pair<double, double> raw_position  = estimate.position;
        double                    confidence     = estimate.confidence;
        int                       best_match_idx = estimate.best_match_idx;

        raw_positions.push_back(raw_position);
        confidences.push_back(confidence);

        // =====================================================================
        // KALMAN UPDATE with recovery mechanism
        // =====================================================================
        std::pair<double, double> filtered_position;
        double innovation      = 0.0;
        bool   outlier_rejected = false;

        if (!kalman_initialized) {
            kalman_filter.initialize(raw_position.first, raw_position.second, 0.0, 0.0);
            kalman_initialized = true;
            filtered_position  = raw_position;
            std::cout << "   ✓ Kalman initialized at frame " << frame_idx << std::endl;
            innovations_history.push_back(0.0);

        } else {
            innovation = kalman_filter.getInnovation(raw_position.first, raw_position.second);

            // For optical flow the displacement per frame is small and smooth:
            // tighten the innovation threshold to catch genuine jumps early.
            double innovation_threshold;
            if (use_optical_flow) {
                // Scale with confidence but keep tighter bounds
                innovation_threshold = (confidence > 0.7) ? 80.0 :
                                       (confidence > 0.5) ? 40.0 : 25.0;
            } else {
                innovation_threshold = (confidence > 0.7) ? 150.0 :
                                       (confidence > 0.5) ?  80.0 : 50.0;
            }

            filtered_position = kalman_filter.update(
                raw_position.first, raw_position.second,
                confidence, innovation_threshold);

            if (innovation > innovation_threshold && confidence < 0.7) {
                outlier_rejected = true;
                ++outliers_rejected;
                ++consecutive_rejections;
                innovations_history.push_back(innovation_threshold);

                if (consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS) {
                    std::cout << "   ⚠️  Filter stuck (" << consecutive_rejections
                              << " consecutive rejections). Resetting to raw estimate." << std::endl;
                    kalman_filter.initialize(
                        raw_position.first, raw_position.second, 0.0, 0.0);
                    filtered_position     = raw_position;
                    consecutive_rejections = 0;
                }
            } else {
                consecutive_rejections = 0;
                innovations_history.push_back(innovation);
            }
        }

        filtered_positions.push_back(filtered_position);
        if (confidence < 0.5) ++low_confidence_count;

        // =====================================================================
        // TELEMETRY
        // =====================================================================
        telemetry_file
            << frame_idx << ","
            << std::fixed << std::setprecision(2) << time_sec << ","
            << std::setprecision(8)
            << raw_position.first      << "," << raw_position.second      << ","
            << predicted_position.first << "," << predicted_position.second << ","
            << filtered_position.first  << "," << filtered_position.second  << ","
            << std::setprecision(4) << confidence << ","
            << std::setprecision(2) << innovation << ","
            << (outlier_rejected ? "1" : "0") << ","
            << best_match_idx << ","
            << algorithm_name << "\n";

        // =====================================================================
        // VISUALIZATION
        // =====================================================================
        cv::Mat display_map = reference_map.clone();

        // Draw starting point
        if (initial_position.first != 0.0 && initial_position.second != 0.0) {
            cv::Point sp = CoordinateUtils::latLngToPixel(
                initial_position.first, initial_position.second,
                center_lat, center_lng, center_x, center_y, mpp,
                meters_per_degree_lat, meters_per_degree_lng);
            cv::circle(display_map, sp, 12, cv::Scalar(203, 105, 255), -1);
            cv::circle(display_map, sp, 14, cv::Scalar(0, 0, 0), 2);
            cv::putText(display_map, "START", cv::Point(sp.x + 15, sp.y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(203, 105, 255), 2);
        }

        // Draw raw path (thin blue)
        for (size_t i = 1; i < raw_positions.size(); ++i) {
            if (raw_positions[i].first == 0.0 || raw_positions[i-1].first == 0.0) continue;
            cv::Point p1 = CoordinateUtils::latLngToPixel(
                raw_positions[i-1].first, raw_positions[i-1].second,
                center_lat, center_lng, center_x, center_y, mpp,
                meters_per_degree_lat, meters_per_degree_lng);
            cv::Point p2 = CoordinateUtils::latLngToPixel(
                raw_positions[i].first, raw_positions[i].second,
                center_lat, center_lng, center_x, center_y, mpp,
                meters_per_degree_lat, meters_per_degree_lng);
            cv::line(display_map, p1, p2, cv::Scalar(255, 0, 0), 1);
        }

        // Draw filtered path (thick, confidence-colour-coded)
        for (size_t i = 1; i < filtered_positions.size(); ++i) {
            cv::Point p1 = CoordinateUtils::latLngToPixel(
                filtered_positions[i-1].first, filtered_positions[i-1].second,
                center_lat, center_lng, center_x, center_y, mpp,
                meters_per_degree_lat, meters_per_degree_lng);
            cv::Point p2 = CoordinateUtils::latLngToPixel(
                filtered_positions[i].first, filtered_positions[i].second,
                center_lat, center_lng, center_x, center_y, mpp,
                meters_per_degree_lat, meters_per_degree_lng);
            double c = confidences[i];
            cv::Scalar col = (c > 0.7) ? cv::Scalar(0,255,0)
                           : (c > 0.5) ? cv::Scalar(0,255,255)
                                       : cv::Scalar(0,0,255);
            cv::line(display_map, p1, p2, col, 2);
        }

        // Draw current position
        if (!filtered_positions.empty()) {
            cv::Point cp = CoordinateUtils::latLngToPixel(
                filtered_position.first, filtered_position.second,
                center_lat, center_lng, center_x, center_y, mpp,
                meters_per_degree_lat, meters_per_degree_lng);
            cv::circle(display_map, cp, 8,  cv::Scalar(0,255,255), -1);
            cv::circle(display_map, cp, 10, cv::Scalar(0,0,0),     2);
        }

        // Overlay text
        std::ostringstream info;
        info << "Frame: " << frame_idx << "/" << total_frames
             << "  t=" << std::fixed << std::setprecision(1) << time_sec
             << "s  Conf=" << std::setprecision(3) << confidence
             << "  Innov=" << std::setprecision(1) << innovation << "m";
        cv::putText(display_map, info.str(),
                    cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    cv::Scalar(255,255,255), 2);

        std::ostringstream pos_info;
        pos_info << "Pos: " << std::fixed << std::setprecision(6)
                 << filtered_position.first << ", " << filtered_position.second;
        cv::putText(display_map, pos_info.str(),
                    cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    cv::Scalar(255,255,255), 2);

        if (outlier_rejected)
            cv::putText(display_map, "OUTLIER REJECTED",
                        cv::Point(10,90), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                        cv::Scalar(0,0,255), 2);
        if (consecutive_rejections > 3)
            cv::putText(display_map,
                        "WARNING: filter drifting (" + std::to_string(consecutive_rejections) + ")",
                        cv::Point(10,120), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                        cv::Scalar(0,140,255), 2);

        if (use_optical_flow) {
            // Show tracked point count
            auto* tracker = dynamic_cast<OpticalFlowTracker*>(estimator.get());
            if (tracker) {
                std::string pt_info = "Tracked pts: " + std::to_string(tracker->trackedPointCount());
                cv::putText(display_map, pt_info,
                            cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                            cv::Scalar(200, 255, 200), 1);
            }
        }

        cv::imshow("Video Navigation", display_map);
        // Always show the original colour frame — processed_frame is internal only
        cv::imshow("Video Frame", frame);

        if (video_writer.isOpened()) {
            cv::Mat out;
            cv::resize(display_map, out, cv::Size(1200, 800));
            video_writer.write(out);
        }

        if (processed_frames % 10 == 0)
            std::cout << "   Frame " << frame_idx << "/" << total_frames
                      << " (" << std::setprecision(1) << (frame_idx * 100.0 / total_frames)
                      << "%)  conf=" << std::setprecision(3) << confidence
                      << "  innov=" << std::setprecision(1) << innovation << "m"
                      << "  consec_reject=" << consecutive_rejections << std::endl;

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) { std::cout << "Cancelled.\n"; break; }
        if (key == ' ')              { std::cout << "Paused.\n"; cv::waitKey(0); }

        ++frame_idx;
        ++processed_frames;
    }

    // =========================================================================
    // CLEANUP & SUMMARY
    // =========================================================================
    video.release();
    if (video_writer.isOpened()) video_writer.release();
    telemetry_file.close();

    std::cout << "\n" << std::string(80,'=') << std::endl;
    std::cout << "✅ COMPLETE  |  " << processed_frames << " frames processed" << std::endl;

    double total_outlier_pct = processed_frames > 0
        ? 100.0 * outliers_rejected / processed_frames : 0.0;
    std::cout << "   Outliers rejected:    " << outliers_rejected
              << " (" << std::setprecision(1) << total_outlier_pct << "%)" << std::endl;
    std::cout << "   Low confidence (<0.5): " << low_confidence_count << std::endl;

    if (!confidences.empty()) {
        double avg = 0; for (double c : confidences) avg += c;
        std::cout << "   Avg confidence: " << std::setprecision(3)
                  << avg / confidences.size() << std::endl;
    }

    std::cout << "   Telemetry: " << telemetry_path << std::endl;

    if (!innovations_history.empty()) {
        cv::Mat graph = createErrorGraph(innovations_history,
                                         algorithm_name + " Kalman Innovation");
        cv::imshow("Tracking Stability", graph);
    }

    std::cout << "\nPress any key to close..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}
