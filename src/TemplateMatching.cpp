#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <memory>
#include "TemplateMatching.hpp"
#include "TemplateMatchingEstimator.hpp"
#include "ORBFeatureEstimator.hpp"
#include "SIFTFeatureEstimator.hpp"
#include "SURFFeatureEstimator.hpp"
#include "HybridEstimator.hpp"
#include "SmoothedEstimator.hpp"
#include "Visualization.hpp"

std::unique_ptr<IPositionEstimator> createPositionEstimator(PositionAlgorithm algorithm) {
    switch(algorithm) {
        case PositionAlgorithm::TEMPLATE:
            return std::make_unique<TemplateMatchingEstimator>();
        
        case PositionAlgorithm::ORB:
            return std::make_unique<ORBFeatureEstimator>(500);
        
        case PositionAlgorithm::SIFT:
            return std::make_unique<SIFTFeatureEstimator>(0);
        
        case PositionAlgorithm::SURF:
            return std::make_unique<SURFFeatureEstimator>(400.0);
        
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
    const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash> &crops)
{
    // Create a copy of the clean map to draw matches on
    cv::Mat display_map = clean_map.clone();

    // Vector to store match results for display
    std::vector<std::tuple<std::pair<double, double>, cv::Point, double>> matches;

    // Iterate through each crop in the unordered_map
    for (const auto &crop_pair : crops)
    {
        const auto &coordinates = crop_pair.first;
        const cv::Mat &cropped_img = crop_pair.second;

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
        for (const auto &crop_pair : crops)
        {
            const cv::Mat &cropped_img = crop_pair.second;
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

// REFACTORED: Consolidated drone simulation function
void runDroneSimulation(
    const cv::Mat &clean_map,
    const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash> &reference_crops,
    const std::vector<std::pair<double, double>> &waypoints,
    double meters_per_degree_lat, double meters_per_degree_lng,
    double center_lat, double center_lng,
    int center_x, int center_y,
    double mpp,
    PositionAlgorithm algorithm)
{
    // Validate input
    if (waypoints.size() < 2) {
        std::cout << "Error: At least 2 waypoints required for navigation" << std::endl;
        return;
    }

    // Create position estimator using factory pattern
    std::unique_ptr<IPositionEstimator> estimator = createPositionEstimator(algorithm);
    std::string algorithm_name = estimator->getName();

    // Determine navigation mode
    bool is_waypoint_mode = waypoints.size() > 2;

    // Initialize simulation parameters
    double speed = 6.0;
    double drift = 0.0;
    double sim_dt = 1.0;
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

    // Setup visualization windows
    cv::namedWindow("Drone Simulation", cv::WINDOW_NORMAL);
    cv::namedWindow("Drone View", cv::WINDOW_NORMAL);
    cv::namedWindow("Telemetry Data", cv::WINDOW_NORMAL);
    cv::resizeWindow("Drone Simulation", 1200, 800);

    // Extract reference crop coordinates
    std::vector<std::pair<double, double>> ref_crop_coords;
    for (const auto &crop_pair : reference_crops) {
        ref_crop_coords.push_back(crop_pair.first);
    }

    // Storage for results and telemetry
    std::vector<std::pair<double, double>> algorithm_positions;
    std::vector<TelemetryEntry> telemetry_entries;
    
    // Create CSV file for telemetry
    std::string telemetry_filename = is_waypoint_mode ? 
        "drone_telemetry_" + algorithm_name + "_waypoints.csv" :
        "drone_telemetry_" + algorithm_name + ".csv";
    std::ofstream telemetry_file(telemetry_filename);
    telemetry_file << "Step,Actual_Lat,Actual_Lng,Estimated_Lat,Estimated_Lng,Match_Confidence,Lat_Error_M,Lng_Error_M,Error_M,Algorithm\n";
    
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
    
    // ==================== MAIN SIMULATION LOOP ====================
    for (int step = 0; step < max_steps; step++)
    {
        // --- Waypoint Navigation Logic ---
        if (is_waypoint_mode && current_waypoint < waypoints.size()) {
            double target_lat = waypoints[current_waypoint].first;
            double target_lng = waypoints[current_waypoint].second;
            
            double dist_to_waypoint_lat = (target_lat - drone.getPosition().first) * meters_per_degree_lat;
            double dist_to_waypoint_lng = (target_lng - drone.getPosition().second) * meters_per_degree_lng;
            double dist_to_waypoint = std::sqrt(dist_to_waypoint_lat * dist_to_waypoint_lat + 
                                              dist_to_waypoint_lng * dist_to_waypoint_lng);
            
            if (dist_to_waypoint < 10.0) {
                current_waypoint++;
                
                if (current_waypoint < waypoints.size()) {
                    double next_lat = waypoints[current_waypoint].first;
                    double next_lng = waypoints[current_waypoint].second;
                    double delta_lng = next_lng - drone.getPosition().second;
                    double delta_lat = next_lat - drone.getPosition().first;
                    double new_heading = std::atan2(delta_lng * meters_per_degree_lng,
                                                delta_lat * meters_per_degree_lat) * 180.0 / M_PI;
                    if (new_heading < 0)
                        new_heading += 360.0;
                    
                    drone.setHeading(new_heading);
                    std::cout << "Reached waypoint " << current_waypoint - 1 
                             << ", heading to waypoint " << current_waypoint << std::endl;
                }
            }
        }

        // --- Update Drone State ---
        drone.step(sim_dt, 0.0);
        drone.updateView(center_lat, center_lng, center_x, center_y, mpp);
        cv::Mat drone_view = drone.getCurrentView();

        // --- Position Estimation (Strategy Pattern) ---
        std::pair<double, double> last_position = algorithm_positions.empty() ? 
            drone.getPosition() : algorithm_positions.back();
        
        PositionEstimate estimate = estimator->estimatePosition(
            drone_view, ref_crop_coords, reference_crops, last_position);
        
        std::pair<double, double> estimated_position = estimate.position;
        double match_confidence = estimate.confidence;
        int best_match_idx = estimate.best_match_idx;

        // --- Calculate Navigation Metrics ---
        const auto &current_actual_pos = drone.getPosition();
        double target_lat = waypoints.back().first;
        double target_lng = waypoints.back().second;
        double dist_to_target_lat = (target_lat - current_actual_pos.first) * meters_per_degree_lat;
        double dist_to_target_lng = (target_lng - current_actual_pos.second) * meters_per_degree_lng;
        double dist_to_target = std::sqrt(dist_to_target_lat * dist_to_target_lat + 
                                         dist_to_target_lng * dist_to_target_lng);
        
        double dist_to_current_waypoint = dist_to_target;
        if (is_waypoint_mode && current_waypoint < waypoints.size()) {
            double wp_lat = waypoints[current_waypoint].first;
            double wp_lng = waypoints[current_waypoint].second;
            double wp_lat_diff = (wp_lat - current_actual_pos.first) * meters_per_degree_lat;
            double wp_lng_diff = (wp_lng - current_actual_pos.second) * meters_per_degree_lng;
            dist_to_current_waypoint = std::sqrt(wp_lat_diff * wp_lat_diff + wp_lng_diff * wp_lng_diff);
        }
        
        // --- Outlier Detection & Correction ---
        if (!algorithm_positions.empty()) {
            double lat_diff = (estimated_position.first - algorithm_positions.back().first) * meters_per_degree_lat;
            double lng_diff = (estimated_position.second - algorithm_positions.back().second) * meters_per_degree_lng;
            double jump_distance = std::sqrt(lat_diff * lat_diff + lng_diff * lng_diff);
            double max_expected_jump = speed * sim_dt * 3.0;
            
            if (jump_distance > max_expected_jump) {
                std::cout << "Warning: Outlier detected at step " << step 
                          << " (jumped " << jump_distance << "m), blending with GPS" << std::endl;
                estimated_position.first = estimated_position.first * 0.3 + current_actual_pos.first * 0.7;
                estimated_position.second = estimated_position.second * 0.3 + current_actual_pos.second * 0.7;
                match_confidence *= 0.4;
            }
        }
        
        // --- Near Waypoint GPS Blending ---
        if (dist_to_current_waypoint < 20.0) {
            double gps_weight = std::min(1.0, (20.0 - dist_to_current_waypoint) / 20.0);
            estimated_position.first = estimated_position.first * (1 - gps_weight) + current_actual_pos.first * gps_weight;
            estimated_position.second = estimated_position.second * (1 - gps_weight) + current_actual_pos.second * gps_weight;
            
            if (is_waypoint_mode && gps_weight > 0.5) {
                std::cout << "Near waypoint " << current_waypoint 
                          << " (" << dist_to_current_waypoint << "m) - GPS weight: " << gps_weight << std::endl;
            }
        }

        // --- Temporal Smoothing ---
        if (!algorithm_positions.empty()) {
            const double smooth_factor = 0.3;
            estimated_position.first = estimated_position.first * (1-smooth_factor) + 
                                   algorithm_positions.back().first * smooth_factor;
            estimated_position.second = estimated_position.second * (1-smooth_factor) + 
                                    algorithm_positions.back().second * smooth_factor;
        }

        // --- Store Results ---
        algorithm_positions.push_back(estimated_position);

        // --- Calculate Errors ---
        double error_lat_m = (current_actual_pos.first - estimated_position.first) * meters_per_degree_lat;
        double error_lng_m = (current_actual_pos.second - estimated_position.second) * meters_per_degree_lng;
        double error_m = std::sqrt(error_lat_m * error_lat_m + error_lng_m * error_lng_m);

        // --- Create Telemetry Entry ---
        TelemetryEntry entry;
        entry.step = step;
        entry.actual_position = current_actual_pos;
        entry.estimated_position = estimated_position;
        entry.confidence = match_confidence;
        entry.lat_error_m = error_lat_m;
        entry.lng_error_m = error_lng_m;
        entry.total_error_m = error_m;
        telemetry_entries.push_back(entry);

        // --- Write to CSV ---
        telemetry_file << step << ","
             << std::fixed << std::setprecision(6) << current_actual_pos.first << ","
             << current_actual_pos.second << ","
             << estimated_position.first << ","
             << estimated_position.second << ","
             << std::setprecision(3) << match_confidence << ","
             << std::setprecision(1) << error_lat_m << ","
             << error_lng_m << ","
             << error_m << ","
             << algorithm_name << "\n";

        // --- Prepare Path Points for Visualization ---
        const auto &flight_path = drone.getFlightPath();
        
        std::vector<cv::Point> gps_path_points;
        for (const auto &position : flight_path) {
            cv::Point pt = drone.latLngToPixel(position.first, position.second,
                                             center_lat, center_lng, center_x, center_y, mpp);
            if (!is_waypoint_mode) pt.x -= 3;
            gps_path_points.push_back(pt);
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_x = std::max(max_x, pt.x);
            max_y = std::max(max_y, pt.y);
        }
        
        std::vector<cv::Point> algo_path_points;
        for (size_t i = 0; i < algorithm_positions.size(); i++) {
            cv::Point pt = drone.latLngToPixel(algorithm_positions[i].first, algorithm_positions[i].second,
                                             center_lat, center_lng, center_x, center_y, mpp);
            if (!is_waypoint_mode) pt.x += 3;
            algo_path_points.push_back(pt);
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_x = std::max(max_x, pt.x);
            max_y = std::max(max_y, pt.y);
        }

        // ==================== VISUALIZATION (REFACTORED) ====================
        
        // Create simulation state
        SimulationState sim_state;
        sim_state.clean_map = clean_map;
        sim_state.waypoint_pixels = waypoint_pixels;
        sim_state.gps_path_points = gps_path_points;
        sim_state.estimated_path_points = algo_path_points;
        sim_state.current_waypoint = current_waypoint;
        sim_state.total_waypoints = waypoints.size();
        sim_state.algorithm_name = algorithm_name;
        sim_state.is_waypoint_mode = is_waypoint_mode;

        // Create telemetry data
        TelemetryData telem_data;
        telem_data.entries = telemetry_entries;
        telem_data.algorithm_name = algorithm_name;
        telem_data.is_waypoint_mode = is_waypoint_mode;

        // Generate visualizations using Visualization module
        cv::Mat sim_vis = createSimulationVisualization(sim_state);
        cv::Mat telem_vis = createTelemetryVisualization(telem_data, step);

        // Apply zoom to simulation view
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
            zoomed_vis = sim_vis(zoom_rect).clone();
        } else {
            zoomed_vis = sim_vis.clone();
        }

        // Update all displays
        updateAllDisplays(zoomed_vis, drone_view, telem_vis);

        // --- Check Termination Conditions ---
        bool destination_reached = false;
        if (is_waypoint_mode) {
            if (current_waypoint >= waypoints.size()) {
                std::cout << "All waypoints reached! Simulation complete." << std::endl;
                destination_reached = true;
            }
        } else {
            if (dist_to_target < 10.0) {
                std::cout << "Destination reached!" << std::endl;
                destination_reached = true;
            }
        }
        
        if (destination_reached) break;

        // Check for user exit
        char key = cv::waitKey(100);
        if (key == 27 || key == 'q') break;
    }
    // ==================== END SIMULATION LOOP ====================
    
    // --- Post-Simulation: Generate and Save Error Graph ---
    cv::Mat error_graph = createErrorGraph(
        drone.getFlightPath(),
        algorithm_positions,
        meters_per_degree_lat,
        meters_per_degree_lng,
        algorithm_name);

    cv::namedWindow("Error Graph", cv::WINDOW_NORMAL);
    cv::resizeWindow("Error Graph", 1000, 600);
    cv::imshow("Error Graph", error_graph);
    cv::waitKey(1);
    
    std::string graph_filename = is_waypoint_mode ?
        "error_graph_" + algorithm_name + "_waypoints.png" :
        "error_graph_" + algorithm_name + ".png";
    
    if (cv::imwrite(graph_filename, error_graph)) {
        std::cout << "Error graph saved to " << graph_filename << std::endl;
    } else {
        std::cout << "Failed to save error graph!" << std::endl;
    }

    telemetry_file.close();
    std::cout << "Telemetry data saved to " << telemetry_filename << std::endl;
    std::cout << "Simulation completed. Press any key to continue..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}
