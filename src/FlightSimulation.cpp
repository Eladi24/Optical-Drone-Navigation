#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <memory>
#include "FlightSimulation.hpp"
#include "TemplateMatchingEstimator.hpp"
#include "ORBFeatureEstimator.hpp"
#include "SIFTFeatureEstimator.hpp"
#include "SURFFeatureEstimator.hpp"
#include "HybridEstimator.hpp"
#include "SmoothedEstimator.hpp"
#include "KalmanFilter.hpp"
#include "Visualization.hpp"
#include "CoordinateUtils.hpp"

std::unique_ptr<IPositionEstimator> createPositionEstimator(PositionAlgorithm algorithm) {
    switch(algorithm) {
        case PositionAlgorithm::TEMPLATE:
            return std::make_unique<TemplateMatchingEstimator>();
        
        case PositionAlgorithm::ORB:
            return std::make_unique<ORBFeatureEstimator>(500);
        
        case PositionAlgorithm::SIFT:
            return std::make_unique<SIFTFeatureEstimator>(0);
        
        case PositionAlgorithm::SURF:
            std::cerr << "Warning: SURF not available, using SIFT instead" << std::endl;
            return std::make_unique<SIFTFeatureEstimator>(0);
        
        case PositionAlgorithm::HYBRID:
            return std::make_unique<HybridEstimator>();
        
        case PositionAlgorithm::SMOOTHED:
            return std::make_unique<SmoothedEstimator>(3);
        
        default:
            return std::make_unique<HybridEstimator>();
    }
}

// Keep matchCropsOnMap as is (helper function for testing)
void matchCropsOnMap(
    const cv::Mat &clean_map,
    const std::vector<ReferenceCrop> &crops)
{
    // Create a copy of the clean map to draw matches on
    cv::Mat display_map = clean_map.clone();

    // Vector to store match results for display
    std::vector<std::tuple<std::pair<double, double>, cv::Point, double>> matches;

    // Iterate through each crop in the unordered_map
    for (const auto &crop : crops)
    {
        const auto &coordinates = crop.coordinates;
        const cv::Mat &cropped_img = crop.image;

        // Get dimensions of the crop
        int width = cropped_img.cols;
        int height = cropped_img.rows;

        // Create result matrix to store matching results
        cv::Mat result;
        int result_cols = clean_map.cols - cropped_img.cols + 1;
        int result_rows = clean_map.rows - cropped_img.rows + 1;
        result.create(result_rows, result_cols, CV_32FC1);

        // Apply template matching
        cv::matchTemplate(clean_map, cropped_img, result, cv::TM_CCOEFF_NORMED);

        // Find the best match location
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        // For TM_CCOEFF_NORMED, the best match is at maxLoc
        cv::Point matchLocation = maxLoc;
        cv::Point matchCenter(matchLocation.x + width / 2, matchLocation.y + height / 2);

        // Store match info
        matches.push_back(std::make_tuple(coordinates, matchLocation, maxVal));

        // Print the coordinates and match confidence
        std::cout << "Crop coordinates: (" << coordinates.first << ", " << coordinates.second << ")" << std::endl;
        std::cout << "Matched location: (" << matchLocation.x << ", " << matchLocation.y << ")" << std::endl;
        std::cout << "Match center: (" << matchCenter.x << ", " << matchCenter.y << ")" << std::endl;
        std::cout << "Match confidence: " << maxVal << std::endl;
        std::cout << "-----------------------------------" << std::endl;

        // Draw rectangle around the matched region with unique colors based on index
        int idx = matches.size() - 1;
        cv::Scalar color(idx * 40 % 255, (idx * 70 + 100) % 255, (idx * 110 + 50) % 255);
        cv::rectangle(display_map, matchLocation,
                      cv::Point(matchLocation.x + cropped_img.cols, matchLocation.y + cropped_img.rows),
                      color, 2);

        // Add text label with confidence score
        std::ostringstream label;
        label << std::fixed << std::setprecision(2) << maxVal;
        cv::putText(display_map, label.str(),
                    cv::Point(matchLocation.x, matchLocation.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }

    // Display results
    cv::namedWindow("Map with Template Matches", cv::WINDOW_NORMAL);
    cv::imshow("Map with Template Matches", display_map);

    // Display individual crops in a grid if there aren't too many
    if (crops.size() <= 25)
    { // Reasonable number to display
        int grid_size = ceil(sqrt(crops.size()));
        int total_width = grid_size * 200; // Assuming crops aren't larger than 200px
        int total_height = grid_size * 200;

        cv::Mat grid(total_height, total_width, CV_8UC3, cv::Scalar(255, 255, 255));

        int i = 0;
        for (const auto &crop : crops)
        {
            const cv::Mat &cropped_img = crop.image;
            int row = i / grid_size;
            int col = i % grid_size;

            int x = col * 200;
            int y = row * 200;

            // Get match info
            const auto &match_info = matches[i];
            cv::Scalar color(std::get<2>(match_info) * 255, 0, 0); // Color based on confidence

            // Create a resized version of the crop if needed
            cv::Mat resized_crop;
            if (cropped_img.cols > 180 || cropped_img.rows > 180)
            {
                double scale = std::min(180.0 / cropped_img.cols, 180.0 / cropped_img.rows);
                cv::resize(cropped_img, resized_crop, cv::Size(), scale, scale);
            }
            else
            {
                resized_crop = cropped_img.clone();
            }

            // Place the crop in the grid
            resized_crop.copyTo(grid(cv::Rect(x + 10, y + 10, resized_crop.cols, resized_crop.rows)));

            // Add a colored border based on match confidence
            cv::rectangle(grid, cv::Rect(x + 9, y + 9, resized_crop.cols + 2, resized_crop.rows + 2),
                          color, 1);

            // Add text label with index
            cv::putText(grid, std::to_string(i),
                        cv::Point(x + 5, y + resized_crop.rows + 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

            i++;
        }

        cv::namedWindow("Crop Templates", cv::WINDOW_NORMAL);
        cv::imshow("Crop Templates", grid);
    }

    std::cout << "Press any key to close windows..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// REFACTORED: Pure Visual Navigation with Kalman Filter
void runDroneSimulation(
    const cv::Mat &clean_map,
    const std::vector<ReferenceCrop> &reference_crops,
    const std::vector<std::pair<double, double>> &waypoints,
    double meters_per_degree_lat, double meters_per_degree_lng,
    double center_lat, double center_lng,
    int center_x, int center_y,
    double mpp,
    PositionAlgorithm algorithm,
    const std::string& location_name,
    const std::string& path_type)
{
    // ==================== INITIALIZATION ====================
    
    // Validate input
    if (waypoints.size() < 2) {
        std::cerr << "Error: At least 2 waypoints required for navigation" << std::endl;
        return;
    }

    if (reference_crops.empty()) {
        std::cerr << "Error: No reference crops provided" << std::endl;
        return;
    }

    // Create position estimator using factory pattern
    std::unique_ptr<IPositionEstimator> estimator = createPositionEstimator(algorithm);
    std::string algorithm_name = estimator->getName();

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "🚁 STARTING KALMAN-FILTERED VISUAL NAVIGATION SIMULATION" << std::endl;
    std::cout << "   Location: " << location_name << std::endl;
    std::cout << "   Path: " << path_type << std::endl;
    std::cout << "   Algorithm: " << algorithm_name << " + Kalman Filter" << std::endl;
    std::cout << "   Reference Crops: " << reference_crops.size() << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Determine navigation mode
    bool is_waypoint_mode = waypoints.size() > 2;

    // Initialize simulation parameters
    double speed = 10.0;                    // m/s
    double drift = 0.0;                    // No drift
    double sim_dt = 1.0;                   // 1 second per step
    int crop_size = static_cast<int>(std::round(100.0 / mpp));
    int max_steps = is_waypoint_mode ? 500 : 100;

    // Calculate initial heading
    double start_lat = waypoints[0].first;
    double start_lng = waypoints[0].second;
    double delta_lng = waypoints[1].second - start_lng;
    double delta_lat = waypoints[1].first - start_lat;
    double initial_heading = std::atan2(delta_lng * meters_per_degree_lng,
                                        delta_lat * meters_per_degree_lat) * 180.0 / M_PI;
    if (initial_heading < 0)
        initial_heading += 360.0;

    // Create drone simulation
    DroneSimulation drone(start_lat, start_lng, initial_heading, speed, drift,
                          meters_per_degree_lat, meters_per_degree_lng,
                          clean_map, crop_size);

    // ==================== KALMAN FILTER INITIALIZATION ====================
    
    // Initialize Kalman filter with tuned parameters
    // process_noise: how much we trust the motion model (lower = more trust)
    // measurement_noise: base noise for visual measurements (higher = less trust)
    DroneKalmanFilter kalman_filter(0.5, 15.0);
    kalman_filter.initialize(start_lat, start_lng, initial_heading, speed);
    
    std::cout << "✓ Kalman Filter initialized" << std::endl;
    std::cout << "  Process Noise: 0.5 (motion model uncertainty)" << std::endl;
    std::cout << "  Measurement Noise: 15.0 (visual measurement base noise)" << std::endl;

    // Setup visualization windows
    cv::namedWindow("Drone Simulation", cv::WINDOW_NORMAL);
    cv::namedWindow("Drone View", cv::WINDOW_NORMAL);
    cv::namedWindow("Telemetry Data", cv::WINDOW_NORMAL);
    cv::resizeWindow("Drone Simulation", 1200, 800);

    // 🆕 VIDEO WRITER SETUP - Simple approach
    std::string video_filename = "Videos/simulation_" + algorithm_name + "_" + 
                                location_name + "_" + path_type + ".avi";
    
    mkdir("Videos", 0755);  // Create directory if needed
    
    int video_width = 1200;
    int video_height = 800;
    double fps = 10.0;  // Matches cv::waitKey(100) = 10fps
    
    cv::VideoWriter video_writer;
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    video_writer.open(video_filename, fourcc, fps, cv::Size(video_width, video_height), true);
    
    if (video_writer.isOpened()) {
        std::cout << "✓ Recording video: " << video_filename << std::endl;
    } else {
        std::cerr << "⚠️  Video recording disabled (could not open writer)" << std::endl;
    }

    // Storage for results and telemetry
    std::vector<std::pair<double, double>> raw_visual_positions;      // Raw from algorithm
    std::vector<std::pair<double, double>> filtered_positions;        // After Kalman filter
    std::vector<std::pair<double, double>> predicted_positions;       // Kalman predictions
    std::vector<TelemetryEntry> telemetry_entries;
    
    // Kalman filter statistics
    int total_outliers_rejected = 0;
    int total_low_confidence_measurements = 0;
    
    // Create CSV file for telemetry
    std::string telemetry_filename = "CSV Files/drone_telemetry_kalman_" + 
                                    algorithm_name + "_" + 
                                    location_name + "_" + 
                                    path_type + ".csv";
    std::ofstream telemetry_file(telemetry_filename);
    if (!telemetry_file.is_open()) {
        std::cerr << "Error: Could not create telemetry file: " << telemetry_filename << std::endl;
        return;
    }
    
    telemetry_file << "Step,Actual_Lat,Actual_Lng,Raw_Lat,Raw_Lng,Predicted_Lat,Predicted_Lng,"
                   << "Filtered_Lat,Filtered_Lng,Match_Confidence,Innovation_M,Outlier_Rejected,"
                   << "Raw_Error_M,Filtered_Error_M,Algorithm\n";
    
    // Path bounding box for zooming
    int min_x = INT_MAX, min_y = INT_MAX, max_x = 0, max_y = 0;
    
    // Calculate pixel coordinates for all waypoints
    std::vector<cv::Point> waypoint_pixels;
    for (const auto &waypoint : waypoints) {
        cv::Point pt = drone.latLngToPixel(waypoint.first, waypoint.second, 
                                         center_lat, center_lng, center_x, center_y, mpp);
        waypoint_pixels.push_back(pt);
        min_x = std::min(min_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_x = std::max(max_x, pt.x);
        max_y = std::max(max_y, pt.y);
    }

    // Current target waypoint
    int current_waypoint = 1;
    
    // Statistics tracking
    int total_large_jumps = 0;
    int total_large_errors = 0;

    std::vector<cv::Point> gps_path_points;
    std::vector<cv::Point> raw_path_points;
    std::vector<cv::Point> filtered_path_points;
    gps_path_points.reserve(max_steps);
    raw_path_points.reserve(max_steps);
    filtered_path_points.reserve(max_steps);
    
    // ==================== MAIN SIMULATION LOOP ====================
    // Add waypoint state tracking
    enum WaypointState { CRUISING, APPROACHING, TURNING };
    WaypointState waypoint_state = CRUISING;

    for (int step = 0; step < max_steps; step++)
    {
        // --- Waypoint Navigation Logic ---
        if (is_waypoint_mode && current_waypoint < waypoints.size()) {
            double target_lat = waypoints[current_waypoint].first;
            double target_lng = waypoints[current_waypoint].second;
            
            double dist_to_waypoint = CoordinateUtils::calculateDistance(
                drone.getPosition(), {target_lat, target_lng},
                meters_per_degree_lat, meters_per_degree_lng);
            
            // State machine for waypoint navigation
            if (dist_to_waypoint > 30.0) {
                waypoint_state = CRUISING;
            } else if (dist_to_waypoint > 10.0) {
                waypoint_state = APPROACHING;
                // Increase process noise when approaching
                kalman_filter.setProcessNoise(2.0);  // 4x normal
            } else {  // dist < 10.0
                waypoint_state = TURNING;
                current_waypoint++;
                
                if (current_waypoint < waypoints.size()) {
                    double next_lat = waypoints[current_waypoint].first;
                    double next_lng = waypoints[current_waypoint].second;
                    double new_heading = CoordinateUtils::calculateHeading(
                        drone.getPosition(), {next_lat, next_lng},
                        meters_per_degree_lat, meters_per_degree_lng);
                    
                    drone.setHeading(new_heading);
                    
                    // 🆕 RESET KALMAN FILTER during turn
                    kalman_filter.initialize(
                        drone.getPosition().first,
                        drone.getPosition().second,
                        new_heading,
                        speed
                    );
                    
                    std::cout << "✓ Waypoint " << current_waypoint - 1 
                             << " reached, turning to " << new_heading << "°" << std::endl;
                }
                
                // Keep high process noise for a few steps after turn
                kalman_filter.setProcessNoise(2.0);
            }
            
            // Reset to normal after stabilizing
            if (waypoint_state == CRUISING) {
                kalman_filter.setProcessNoise(0.5);  // Back to normal
            }
        }

        // --- Update Drone State ---
        drone.step(sim_dt, 0.0);
        drone.updateView(center_lat, center_lng, center_x, center_y, mpp);
        const cv::Mat& drone_view = drone.getCurrentView();

        if (drone_view.empty()) {
            std::cerr << "Error: Drone view is empty at step " << step << std::endl;
            break;
        }

        // ==================== KALMAN FILTER PREDICTION ====================
        
        // Predict position based on motion model
        std::pair<double, double> predicted_position = kalman_filter.predict(sim_dt);
        predicted_positions.push_back(predicted_position);

        // ==================== VISUAL MEASUREMENT ====================
        
        // Get last filtered position (or use prediction if first step)
        std::pair<double, double> last_position = filtered_positions.empty() ? 
            predicted_position : filtered_positions.back();
        
        // Get raw visual estimate
        PositionEstimate estimate = estimator->estimatePosition(
            drone_view, reference_crops, last_position);
        
        std::pair<double, double> raw_visual_position = estimate.position;
        double match_confidence = estimate.confidence;
        int best_match_idx = estimate.best_match_idx;
        
        raw_visual_positions.push_back(raw_visual_position);

        // 🆕 GET MATCHED REFERENCE CROP FOR VISUALIZATION
        cv::Mat matched_crop;
        if (best_match_idx >= 0 && best_match_idx < reference_crops.size()) {
            matched_crop = reference_crops[best_match_idx].image.clone();
        }

        // ==================== KALMAN FILTER UPDATE ====================
        
        // Calculate innovation (difference between measurement and prediction)
        double innovation = kalman_filter.getInnovation(
            raw_visual_position.first, raw_visual_position.second);
        
        // Adaptive innovation threshold based on confidence
        double expected_distance = speed * sim_dt;
        double base_threshold = 50.0;
        double confidence_factor = (1.0 - match_confidence) * 2.0;  // Lower conf → higher threshold
        double motion_factor = 1.0;

        // 🆕 Check if near waypoint (allow larger deviations during turns)
        if (is_waypoint_mode && current_waypoint < waypoints.size()) {
            double dist_to_next_waypoint = CoordinateUtils::calculateDistance(
                drone.getPosition(), 
                {waypoints[current_waypoint].first, waypoints[current_waypoint].second},
                meters_per_degree_lat, meters_per_degree_lng);
            
            if (dist_to_next_waypoint < 30.0) {  // Within 30m of waypoint
                motion_factor = 3.0;  // Allow 3x larger innovation during approach
                std::cout << "⚠️ Near waypoint " << current_waypoint 
                          << " - relaxing innovation threshold" << std::endl;
            }
        }

        double innovation_threshold;
        switch(waypoint_state) {
            case CRUISING:
                innovation_threshold = match_confidence > 0.7 ? 50.0 : 30.0;
                break;
            case APPROACHING:
                innovation_threshold = 100.0;  // More permissive
                break;
            case TURNING:
                innovation_threshold = 150.0;  // Very permissive during turn
                break;
        }
        
        // Update Kalman filter with visual measurement
        std::pair<double, double> filtered_position = kalman_filter.update(
            raw_visual_position.first, 
            raw_visual_position.second,
            match_confidence,
            innovation_threshold
        );
        
        filtered_positions.push_back(filtered_position);
        
        // Check if measurement was rejected
        bool outlier_rejected = false;
        if (innovation > innovation_threshold && match_confidence < 0.7) {
            outlier_rejected = true;
            total_outliers_rejected++;
            std::cout << "\n🔍 Kalman Filter: Outlier rejected at step " << step << std::endl;
            std::cout << "   Innovation: " << std::fixed << std::setprecision(1) 
                      << innovation << "m (threshold: " << innovation_threshold << "m)" << std::endl;
            std::cout << "   Confidence: " << std::setprecision(3) << match_confidence << std::endl;
            std::cout << "   Using predicted position instead" << std::endl;
        }
        
        if (match_confidence < 0.5) {
            total_low_confidence_measurements++;
        }

        // Get current GPS position (FOR ERROR CALCULATION ONLY)
        const auto &current_actual_pos = drone.getPosition();

        // ==================== ERROR ANALYSIS ====================
        
        // Calculate errors for both raw and filtered positions
        double raw_error_lat = (current_actual_pos.first - raw_visual_position.first) * meters_per_degree_lat;
        double raw_error_lng = (current_actual_pos.second - raw_visual_position.second) * meters_per_degree_lng;
        double raw_error_m = std::sqrt(raw_error_lat * raw_error_lat + raw_error_lng * raw_error_lng);
        
        double filtered_error_lat = (current_actual_pos.first - filtered_position.first) * meters_per_degree_lat;
        double filtered_error_lng = (current_actual_pos.second - filtered_position.second) * meters_per_degree_lng;
        double filtered_error_m = std::sqrt(filtered_error_lat * filtered_error_lat + 
                                           filtered_error_lng * filtered_error_lng);
        
        // Detect large jumps in raw visual measurements
        if (!raw_visual_positions.empty() && raw_visual_positions.size() > 1) {
            size_t prev_idx = raw_visual_positions.size() - 2;
            double lat_diff = (raw_visual_position.first - raw_visual_positions[prev_idx].first) * meters_per_degree_lat;
            double lng_diff = (raw_visual_position.second - raw_visual_positions[prev_idx].second) * meters_per_degree_lng;
            double jump_distance = std::sqrt(lat_diff * lat_diff + lng_diff * lng_diff);
            double max_expected_jump = speed * sim_dt * 3.0;
            
            if (jump_distance > max_expected_jump) {
                total_large_jumps++;
                std::cout << "\n⚠️  Large raw visual jump at step " << step << std::endl;
                std::cout << "   Jump: " << std::fixed << std::setprecision(1) << jump_distance 
                          << "m (max: " << max_expected_jump << "m)" << std::endl;
                std::cout << "   Kalman innovation: " << innovation << "m" << std::endl;
                std::cout << "   " << (outlier_rejected ? "✓ Rejected by filter" : "⚠️ Accepted by filter") << std::endl;
            }
        }
        
        // Log large filtered errors
        if (filtered_error_m > 50.0) {
            total_large_errors++;
            std::cout << "\n❌ LARGE FILTERED ERROR at step " << step << ": " 
                      << std::fixed << std::setprecision(1) << filtered_error_m << "m" << std::endl;
            std::cout << "   Raw error: " << raw_error_m << "m" << std::endl;
            std::cout << "   Confidence: " << std::setprecision(3) << match_confidence << std::endl;
            std::cout << "   Innovation: " << std::setprecision(1) << innovation << "m" << std::endl;
            
            // Check distance to nearest reference crop
            double min_crop_dist = 1e10;
            for (const auto& crop : reference_crops) {
                double dist = CoordinateUtils::calculateDistance(
                    current_actual_pos, crop.coordinates,
                    meters_per_degree_lat, meters_per_degree_lng);
                min_crop_dist = std::min(min_crop_dist, dist);
            }
            std::cout << "   Nearest crop: " << min_crop_dist << "m" << std::endl;
        }

        // ==================== TELEMETRY ====================
        
        // Create telemetry entry
        TelemetryEntry entry;
        entry.step = step;
        entry.actual_position = current_actual_pos;
        entry.estimated_position = filtered_position;  // Use filtered position
        entry.confidence = match_confidence;
        entry.lat_error_m = filtered_error_lat;
        entry.lng_error_m = filtered_error_lng;
        entry.total_error_m = filtered_error_m;
        telemetry_entries.push_back(entry);

        // Write to CSV
        telemetry_file << step << ","
             << std::fixed << std::setprecision(6) 
             << current_actual_pos.first << "," << current_actual_pos.second << ","
             << raw_visual_position.first << "," << raw_visual_position.second << ","
             << predicted_position.first << "," << predicted_position.second << ","
             << filtered_position.first << "," << filtered_position.second << ","
             << std::setprecision(3) << match_confidence << ","
             << std::setprecision(1) << innovation << ","
             << (outlier_rejected ? "1" : "0") << ","
             << raw_error_m << "," << filtered_error_m << ","
             << algorithm_name << "\n";

        // ==================== VISUALIZATION ====================
        
        // Prepare path points
        const auto &flight_path = drone.getFlightPath();
        
        gps_path_points.clear();
        for (const auto &position : flight_path) {
            cv::Point pt = drone.latLngToPixel(position.first, position.second,
                                             center_lat, center_lng, center_x, center_y, mpp);
            gps_path_points.push_back(pt);
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_x = std::max(max_x, pt.x);
            max_y = std::max(max_y, pt.y);
        }
        
        // Raw visual path (for comparison)
        raw_path_points.clear();
        for (const auto& pos : raw_visual_positions) {
            cv::Point pt = drone.latLngToPixel(pos.first, pos.second,
                                             center_lat, center_lng, center_x, center_y, mpp);
            raw_path_points.push_back(pt);
        }
        
        // Filtered path (main result)
        filtered_path_points.clear();
        for (const auto& pos : filtered_positions) {
            cv::Point pt = drone.latLngToPixel(pos.first, pos.second,
                                             center_lat, center_lng, center_x, center_y, mpp);
            filtered_path_points.push_back(pt);
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_x = std::max(max_x, pt.x);
            max_y = std::max(max_y, pt.y);
        }

        // Create simulation state
        SimulationState sim_state;
        sim_state.clean_map = clean_map;
        sim_state.waypoint_pixels = waypoint_pixels;
        sim_state.gps_path_points = gps_path_points;
        sim_state.estimated_path_points = filtered_path_points;  // Use filtered
        sim_state.current_waypoint = current_waypoint;
        sim_state.total_waypoints = waypoints.size();
        sim_state.algorithm_name = algorithm_name + " + Kalman";
        sim_state.is_waypoint_mode = is_waypoint_mode;

        // Create telemetry data
        TelemetryData telem_data;
        telem_data.entries = telemetry_entries;
        telem_data.algorithm_name = algorithm_name + " + Kalman";
        telem_data.is_waypoint_mode = is_waypoint_mode;

        // Generate visualizations
        cv::Mat sim_vis = createSimulationVisualization(sim_state);
        
        // Optionally draw raw visual path in gray for comparison
        for (size_t i = 1; i < raw_path_points.size(); i++) {
            cv::line(sim_vis, raw_path_points[i-1], raw_path_points[i], 
                    cv::Scalar(150, 150, 150), 1, cv::LINE_AA);
        }
        
        cv::Mat telem_vis = createTelemetryVisualization(telem_data, step);

        // Apply zoom FIRST
        int width = max_x - min_x;
        int height = max_y - min_y;
        int padding_x = std::max(static_cast<int>(width * 0.2), 50);
        int padding_y = std::max(static_cast<int>(height * 0.2), 50);
        
        cv::Rect zoom_rect(
            std::max(0, min_x - padding_x),
            std::max(0, min_y - padding_y),
            std::min(clean_map.cols - std::max(0, min_x - padding_x), width + 2 * padding_x),
            std::min(clean_map.rows - std::max(0, min_y - padding_y), height + 2 * padding_y)
        );
        zoom_rect = zoom_rect & cv::Rect(0, 0, clean_map.cols, clean_map.rows);
        
        cv::Mat zoomed_vis;
        if (zoom_rect.width > 0 && zoom_rect.height > 0) {
            zoomed_vis = sim_vis(zoom_rect);
        } else {
            zoomed_vis = sim_vis;
        }

        // Add legend to zoomed visualization
        double current_error = telemetry_entries.empty() ? -1.0 : telemetry_entries.back().total_error_m;
        cv::Mat canvas = addLegendToCanvas(
            zoomed_vis,
            is_waypoint_mode,
            algorithm_name + " + Kalman",
            speed,
            drone.getHeading(),
            sim_dt,
            crop_size,
            current_error,
            drone_view,          // 🆕 Pass current drone view
            matched_crop         // 🆕 Pass matched reference crop
        );

        // NOW use canvas for display and video
        cv::Mat video_frame;
        cv::resize(canvas, video_frame, cv::Size(video_width, video_height));
        if (video_writer.isOpened()) video_writer.write(video_frame);
        
        // Update displays
        updateAllDisplays(canvas, drone_view, telem_vis);
        
        // 🆕 PAUSE FOR 2 SECONDS TO SHOW THE MATCH
        for (int pause_frame = 0; pause_frame < 3; pause_frame++) {
            video_writer.write(video_frame);
        }
        cv::waitKey(300);
        
        // --- Check Termination Conditions ---
        bool destination_reached = false;
        if (is_waypoint_mode) {
            if (current_waypoint >= waypoints.size()) {
                std::cout << "\n✓ All waypoints reached! Simulation complete." << std::endl;
                destination_reached = true;
            }
        } else {
            double target_lat = waypoints.back().first;
            double target_lng = waypoints.back().second;
            double dist = CoordinateUtils::calculateDistance(
                drone.getPosition(), {target_lat, target_lng},
                meters_per_degree_lat, meters_per_degree_lng);
            if (dist < 10.0) {
                std::cout << "\n✓ Destination reached!" << std::endl;
                destination_reached = true;
            }
        }
        
        if (destination_reached) break;

        // // Check for user exit
        // char key = cv::waitKey(100);
        // if (key == 27 || key == 'q') {
        //     std::cout << "\nSimulation terminated by user." << std::endl;
        //     break;
        // }
    }
    // ==================== END SIMULATION LOOP ====================
    
    // ==================== POST-SIMULATION STATISTICS ====================
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "📊 KALMAN FILTER SIMULATION STATISTICS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Total steps: " << telemetry_entries.size() << std::endl;
    std::cout << "Raw visual large jumps: " << total_large_jumps << std::endl;
    std::cout << "Outliers rejected by Kalman: " << total_outliers_rejected << std::endl;
    std::cout << "Low confidence measurements: " << total_low_confidence_measurements << std::endl;
    std::cout << "Large filtered errors (>50m): " << total_large_errors << std::endl;
    
    // Calculate error statistics
    if (!telemetry_entries.empty()) {
        double sum_raw_error = 0.0;
        double sum_filtered_error = 0.0;
        double max_raw_error = 0.0;
        double max_filtered_error = 0.0;
        double min_filtered_error = 1e10;
        
        for (size_t i = 0; i < telemetry_entries.size(); i++) {
            const auto& entry = telemetry_entries[i];
            const auto& raw_pos = raw_visual_positions[i];
            const auto& actual_pos = entry.actual_position;
            
            double raw_lat_err = (actual_pos.first - raw_pos.first) * meters_per_degree_lat;
            double raw_lng_err = (actual_pos.second - raw_pos.second) * meters_per_degree_lng;
            double raw_err = std::sqrt(raw_lat_err * raw_lat_err + raw_lng_err * raw_lng_err);
            
            sum_raw_error += raw_err;
            sum_filtered_error += entry.total_error_m;
            max_raw_error = std::max(max_raw_error, raw_err);
            max_filtered_error = std::max(max_filtered_error, entry.total_error_m);
            min_filtered_error = std::min(min_filtered_error, entry.total_error_m);
        }
        
        double mean_raw_error = sum_raw_error / telemetry_entries.size();
        double mean_filtered_error = sum_filtered_error / telemetry_entries.size();
        double improvement = ((mean_raw_error - mean_filtered_error) / mean_raw_error) * 100.0;
        
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "\n--- Raw Visual Estimates ---" << std::endl;
        std::cout << "Mean error: " << mean_raw_error << "m" << std::endl;
        std::cout << "Max error: " << max_raw_error << "m" << std::endl;
        
        std::cout << "\n--- Kalman Filtered Estimates ---" << std::endl;
        std::cout << "Mean error: " << mean_filtered_error << "m" << std::endl;
        std::cout << "Max error: " << max_filtered_error << "m" << std::endl;
        std::cout << "Min error: " << min_filtered_error << "m" << std::endl;
        
        std::cout << "\n--- Improvement ---" << std::endl;
        std::cout << "Error reduction: " << improvement << "%" << std::endl;
        std::cout << "Outlier rejection rate: " << std::setprecision(1) 
                  << (100.0 * total_outliers_rejected / telemetry_entries.size()) << "%" << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;
    
    // 🆕 ADD ERROR GRAPH TO VIDEO (5 seconds)
    if (video_writer.isOpened()) {
        std::cout << "\nAdding error graph to video (5 seconds)..." << std::endl;
        
        // Generate error graph first
        std::vector<double> error_values;
        error_values.reserve(telemetry_entries.size());
        for (const auto& entry : telemetry_entries) {
            error_values.push_back(entry.total_error_m);
        }
        
        cv::Mat error_graph = createErrorGraph(error_values, algorithm_name + " + Kalman");
        
        // Resize to video dimensions
        cv::Mat graph_resized;
        cv::resize(error_graph, graph_resized, cv::Size(video_width, video_height));
        
        // Write 50 frames (5 seconds at 10fps)
        for (int i = 0; i < 50; i++) {
            video_writer.write(graph_resized);
        }
        
        video_writer.release();
        std::cout << "✓ Video saved: " << video_filename << std::endl;
    }

    // --- Generate and Save Error Comparison Graph ---
    // 🆕 Extract errors from telemetry (already calculated correctly)
    std::vector<double> error_values;
    error_values.reserve(telemetry_entries.size());
    for (const auto& entry : telemetry_entries) {
        error_values.push_back(entry.total_error_m);
    }

    std::cout << "\nGenerating error graph with " << error_values.size() << " data points..." << std::endl;

    cv::Mat error_graph = createErrorGraph(
        error_values,  // ✅ Pass pre-calculated errors
        algorithm_name + " + Kalman");

    cv::namedWindow("Error Graph", cv::WINDOW_NORMAL);
    cv::resizeWindow("Error Graph", 1000, 600);
    cv::imshow("Error Graph", error_graph);
    cv::waitKey(1);

    std::string graph_filename = "Images/error_graph_kalman_" + 
                                algorithm_name + "_" + 
                                location_name + "_" + 
                                path_type + ".png";

    if (cv::imwrite(graph_filename, error_graph)) {
        std::cout << "✓ Error graph saved to: " << graph_filename << std::endl;
    } else {
        std::cerr << "✗ Failed to save error graph!" << std::endl;
    }

    telemetry_file.close();
    std::cout << "✓ Telemetry data saved to: " << telemetry_filename << std::endl;
    std::cout << "\nSimulation completed. Press any key to continue..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}
