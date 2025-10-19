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
#include "TemplateMatching.hpp"

// New function for template matching with map crops
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

void DroneSimulation::step(double dt_seconds, double correction_degrees)
{
    // Apply random drift plus any correction to heading
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> drift_dist(0.0, drift);

    heading += drift_dist(gen);
    heading += correction_degrees;

    // Normalize heading to 0-360 degrees
    while (heading < 0)
        heading += 360;
    while (heading >= 360)
        heading -= 360;

    // Calculate movement in lat/lng based on heading and speed
    double distance_m = speed * dt_seconds;
    double heading_rad = heading * M_PI / 180.0;

    double north_m = distance_m * cos(heading_rad);
    double east_m = distance_m * sin(heading_rad);

    double lat_change = north_m / meters_per_degree_lat;
    double lng_change = east_m / meters_per_degree_lng;

    // Update position
    lat += lat_change;
    lng += lng_change;

    // Record new position
    flight_path.push_back(std::make_pair(lat, lng));
}

void DroneSimulation::updateView(double center_lat, double center_lng, int center_x, int center_y, double mpp)
{
    // Convert drone position to pixel coordinates
    cv::Point drone_pos = latLngToPixel(lat, lng, center_lat, center_lng, center_x, center_y, mpp);

    // Calculate crop region for drone view
    cv::Rect view_rect(
        drone_pos.x - view_size / 2,
        drone_pos.y - view_size / 2,
        view_size,
        view_size);

    // Ensure crop region is within map bounds
    view_rect = view_rect & cv::Rect(0, 0, map.cols, map.rows);

    // Update drone's view if valid
    if (view_rect.width > 0 && view_rect.height > 0)
    {
        drone_view = map(view_rect).clone();
    }
}

cv::Point DroneSimulation::latLngToPixel(double lat, double lng, double center_lat, double center_lng,
                                         int center_x, int center_y, double mpp)
{
    double lat_diff = lat - center_lat; // Difference in latitude
    double lng_diff = lng - center_lng; // Difference in longitude

    int y_offset = static_cast<int>(-(lat_diff * meters_per_degree_lat) / mpp);
    int x_offset = static_cast<int>((lng_diff * meters_per_degree_lng) / mpp);

    return cv::Point(center_x + x_offset, center_y + y_offset);
}

// Function to estimate drone position from camera view using image matching
std::tuple<std::pair<double, double>, double, int> estimatePositionFromImage(
    const cv::Mat &drone_view,
    const std::vector<std::pair<double, double>> &ref_crop_coords,
    const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash> &reference_crops)
{
    // Vector to store match scores for each reference image
    std::vector<std::pair<int, double>> all_scores;

    // Calculate match score for each reference crop
    for (size_t idx = 0; idx < ref_crop_coords.size(); idx++)
    {
        const auto &crop_pair = *std::next(reference_crops.begin(), idx);
        const cv::Mat &ref_crop = crop_pair.second;

        // Ensure comparable sizes
        cv::Mat comparison_image;
        if (drone_view.size() != ref_crop.size())
        {
            cv::resize(ref_crop, comparison_image, drone_view.size());
        }
        else
        {
            comparison_image = ref_crop;
        }

        // Calculate match score
        cv::Mat result;
        cv::matchTemplate(comparison_image, drone_view, result, cv::TM_CCOEFF_NORMED);

        double score;
        cv::minMaxLoc(result, nullptr, &score);
        all_scores.push_back(std::make_pair(idx, score));
    }

    // Sort scores in descending order
    std::sort(all_scores.begin(), all_scores.end(),
              [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
                  return a.second > b.second;
              });

    // Get best match
    int best_match_idx = all_scores.empty() ? -1 : all_scores[0].first;
    double match_confidence = all_scores.empty() ? 0.0 : all_scores[0].second;

    // Get estimated position (or default to {0,0} if no match)
    std::pair<double, double> estimated_position;
    if (best_match_idx >= 0)
    {
        estimated_position = ref_crop_coords[best_match_idx];
    }
    else
    {
        estimated_position = {0.0, 0.0};
    }

    return std::make_tuple(estimated_position, match_confidence, best_match_idx);
}

// Function to create visualization of match scores
cv::Mat createScoreVisualization(const std::vector<std::pair<int, double>> &all_scores, int total_crops)
{
    cv::Mat score_viz(200, total_crops * 40 + 20, CV_8UC3, cv::Scalar(240, 240, 240));

    // Draw bars representing match scores
    for (size_t i = 0; i < all_scores.size(); i++)
    {
        int idx = all_scores[i].first;
        double score = all_scores[i].second;
        int bar_height = static_cast<int>(score * 150); // Scale to max height of 150px

        // Color based on confidence
        cv::Scalar bar_color;
        if (score < 0.4)
            bar_color = cv::Scalar(0, 0, 255);      // Red
        else if (score < 0.6)
            bar_color = cv::Scalar(0, 255, 255); // Yellow
        else
            bar_color = cv::Scalar(0, 255, 0);        // Green

        // Draw the bar
        cv::rectangle(score_viz,
                      cv::Point(20 + i * 40, 180),
                      cv::Point(50 + i * 40, 180 - bar_height),
                      bar_color, -1);

        // Add score text
        std::ostringstream score_text;
        score_text << std::fixed << std::setprecision(2) << score;
        cv::putText(score_viz, score_text.str(),
                    cv::Point(20 + i * 40, 195),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

        // Add index number
        cv::putText(score_viz, std::to_string(idx),
                    cv::Point(20 + i * 40, 180 - bar_height - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1);
    }

    // Draw baseline
    cv::line(score_viz, cv::Point(10, 180), cv::Point(total_crops * 40 + 10, 180),
             cv::Scalar(0, 0, 0), 1);

    return score_viz;
}

// Function to update reference grid visualization based on match scores
cv::Mat updateReferenceGridVisualization(
    const cv::Mat &ref_grid,
    const std::vector<std::pair<int, double>> &all_scores,
    int best_match_idx,
    int grid_size)
{
    cv::Mat step_grid = ref_grid.clone();

    // Highlight borders based on match scores
    for (const auto &score_pair : all_scores)
    {
        int idx = score_pair.first;
        double score = score_pair.second;

        int row = idx / grid_size;
        int col = idx % grid_size;
        int x = col * 200;
        int y = row * 200;

        // Color based on score (red→yellow→green)
        cv::Scalar border_color;
        if (score < 0.4)
            border_color = cv::Scalar(0, 0, 255);      // Red
        else if (score < 0.6)
            border_color = cv::Scalar(0, 255, 255); // Yellow
        else
            border_color = cv::Scalar(0, 255, 0);        // Green

        // Draw colored border around the reference image
        cv::Rect image_area(x + 9, y + 9, 182, 182);
        cv::rectangle(step_grid, image_area, border_color, 1);

        // Add score text
        std::ostringstream score_text;
        score_text << std::fixed << std::setprecision(2) << score;
        cv::putText(step_grid, score_text.str(),
                    cv::Point(x + 50, y + 160),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1);
    }

    // Highlight the best match if one exists
    if (best_match_idx >= 0)
    {
        int row = best_match_idx / grid_size;
        int col = best_match_idx % grid_size;
        int x = col * 200;
        int y = row * 200;

        cv::Scalar highlight_color(0, 255, 255); // Yellow highlight
        cv::Rect image_area(x + 7, y + 7, 186, 186); // Slightly larger for emphasis
        cv::rectangle(step_grid, image_area, highlight_color, 3);
    }

    return step_grid;
}

// Function to run the drone simulation
void runDroneSimulation(
    const cv::Mat &clean_map,
    const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash> &reference_crops,
    double start_lat, double start_lng,
    double end_lat, double end_lng,
    double meters_per_degree_lat, double meters_per_degree_lng,
    double center_lat, double center_lng,
    int center_x, int center_y,
    double mpp)
{
    // Calculate initial heading from start to end
    double delta_lng = end_lng - start_lng;
    double delta_lat = end_lat - start_lat;
    double initial_heading = std::atan2(delta_lng * meters_per_degree_lng,
                                        delta_lat * meters_per_degree_lat) *
                             180.0 / M_PI;
    if (initial_heading < 0)
        initial_heading += 360.0;

    // Initialize simulation parameters
    double speed = 6.0;                                        // m/s
    double drift = 0.0;                                        // Set to zero for accurate path following
    double sim_dt = 1.0;                                       // time step in seconds
    int crop_size = static_cast<int>(std::round(100.0 / mpp)); // 100m view size
    int max_steps = 100;

    // Create drone simulation - flying mainly by GPS but with some drift
    DroneSimulation drone(start_lat, start_lng, initial_heading, speed, drift,
                          meters_per_degree_lat, meters_per_degree_lng,
                          clean_map, crop_size);

    // Setup visualization - only keep necessary windows
    cv::namedWindow("Drone Simulation", cv::WINDOW_NORMAL);
    cv::namedWindow("Drone View", cv::WINDOW_NORMAL);
    cv::namedWindow("Telemetry Data", cv::WINDOW_NORMAL);
    
    // Set the Drone Simulation window to be larger
    cv::resizeWindow("Drone Simulation", 1200, 800);

    // Extract reference crop coordinates into a vector for easier access
    std::vector<std::pair<double, double>> ref_crop_coords;
    for (const auto &crop_pair : reference_crops) {
        ref_crop_coords.push_back(crop_pair.first);
    }

    // Store algorithm estimated positions and match confidences
    std::vector<std::pair<double, double>> algorithm_positions;
    std::vector<double> match_confidences;
    
    // Create CSV file for telemetry data
    std::ofstream telemetry_file("drone_telemetry.csv");
    telemetry_file << "Step,Actual_Lat,Actual_Lng,Estimated_Lat,Estimated_Lng,Match_Confidence,Lat_Error_M,Lng_Error_M,Error_M\n";
    
    // Create a telemetry data visualization matrix
    int telemetry_height = 600;
    int telemetry_width = 800;
    cv::Mat telemetry_vis(telemetry_height, telemetry_width, CV_8UC3, cv::Scalar(240, 240, 240));
    
    // Variables for path bounding box to enable zooming
    int min_x = INT_MAX, min_y = INT_MAX, max_x = 0, max_y = 0;
    cv::Point start_pt = drone.latLngToPixel(start_lat, start_lng, 
                                          center_lat, center_lng, center_x, center_y, mpp);
    cv::Point end_pt = drone.latLngToPixel(end_lat, end_lng,
                                        center_lat, center_lng, center_x, center_y, mpp);
    
    // Initialize bounding box with start/end points
    min_x = std::min(min_x, std::min(start_pt.x, end_pt.x));
    min_y = std::min(min_y, std::min(start_pt.y, end_pt.y));
    max_x = std::max(max_x, std::max(start_pt.x, end_pt.x));
    max_y = std::max(max_y, std::max(start_pt.y, end_pt.y));

    // Main simulation loop
    for (int step = 0; step < max_steps; step++)
    {
        // 1. Update drone's position based on GPS (with simulated drift)
        drone.step(sim_dt, 0.0);

        // 2. Update drone's current view (image capture)
        drone.updateView(center_lat, center_lng, center_x, center_y, mpp);
        cv::Mat drone_view = drone.getCurrentView();

        // 3. Use hybrid image matching to estimate position
        std::pair<double, double> last_position = algorithm_positions.empty() ? 
            drone.getPosition() : algorithm_positions.back();
            
        auto [estimated_position, match_confidence, best_match_idx] =
            estimatePositionHybrid(drone_view, ref_crop_coords, reference_crops, last_position);

        // Add temporal smoothing to reduce sudden jumps
        if (!algorithm_positions.empty()) {
            const double smooth_factor = 0.3; // 0-1, higher means more smoothing
            estimated_position.first = estimated_position.first * (1-smooth_factor) + 
                                   algorithm_positions.back().first * smooth_factor;
            estimated_position.second = estimated_position.second * (1-smooth_factor) + 
                                    algorithm_positions.back().second * smooth_factor;
        }

        // Always store the estimated position regardless of match quality
        algorithm_positions.push_back(estimated_position);
        match_confidences.push_back(match_confidence);

        // Write telemetry data to CSV
        double error_lat_m = (drone.getPosition().first - estimated_position.first) * meters_per_degree_lat;
        double error_lng_m = (drone.getPosition().second - estimated_position.second) * meters_per_degree_lng;
        double error_m = std::sqrt(error_lat_m * error_lat_m + error_lng_m * error_lng_m);

        telemetry_file << step << ","
             << std::fixed << std::setprecision(6) << drone.getPosition().first << ","
             << drone.getPosition().second << ","
             << estimated_position.first << ","
             << estimated_position.second << ","
             << std::setprecision(3) << match_confidence << ","
             << std::setprecision(1) << error_lat_m << ","  // Add lat error
             << error_lng_m << ","                         // Add lng error
             << error_m << "\n";

        // Create visualization for this step
        cv::Mat step_vis = clean_map.clone();

        // Get the flight path so far
        const auto &flight_path = drone.getFlightPath();

        // Collect GPS path points (actual drone path)
        std::vector<cv::Point> gps_path_points;
        for (const auto &position : flight_path) {
            cv::Point pt = drone.latLngToPixel(position.first, position.second,
                                             center_lat, center_lng, center_x, center_y, mpp);
            // Add a small offset to the left for better visibility
            pt.x -= 3;
            gps_path_points.push_back(pt);
            
            // Update bounding box for zoom
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_x = std::max(max_x, pt.x);
            max_y = std::max(max_y, pt.y);
        }
        
        // Collect algorithm path points (all estimated positions)
        std::vector<cv::Point> algo_path_points;
        for (size_t i = 0; i < algorithm_positions.size(); i++) {
            // Convert algorithm's lat/lng estimate to pixel coordinates
            cv::Point pt = drone.latLngToPixel(algorithm_positions[i].first, algorithm_positions[i].second,
                                             center_lat, center_lng, center_x, center_y, mpp);
            
            // Add a small offset to the right for better visibility
            pt.x += 3;
            algo_path_points.push_back(pt);
            
            // Update bounding box for zoom
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_x = std::max(max_x, pt.x);
            max_y = std::max(max_y, pt.y);
        }
        
        // Draw the GPS-based path (actual path) - orange (slightly thicker)
        for (size_t i = 1; i < gps_path_points.size(); i++) {
            cv::line(step_vis, gps_path_points[i-1], gps_path_points[i], 
                    cv::Scalar(0, 165, 255), 5);    // Orange line for GPS path
        }

        // Draw the algorithm-based path (estimated positions) - yellow
        for (size_t i = 1; i < algo_path_points.size(); i++) {
            // Use consistent yellow color for estimated path
            cv::line(step_vis, algo_path_points[i-1], algo_path_points[i], 
                    cv::Scalar(0, 255, 255), 4);    // Yellow line for estimated path
        }

        // Draw markers for current positions
        // Current GPS position
        if (!gps_path_points.empty()) {
            cv::circle(step_vis, gps_path_points.back(), 6, cv::Scalar(0, 165, 255), -1);
            cv::circle(step_vis, gps_path_points.back(), 6, cv::Scalar(0, 0, 0), 1); // Black outline
        }
        
        // Current algorithm position
        if (!algo_path_points.empty()) {
            cv::circle(step_vis, algo_path_points.back(), 6, cv::Scalar(0, 255, 255), -1); // Yellow
            cv::circle(step_vis, algo_path_points.back(), 6, cv::Scalar(0, 0, 0), 1); // Black outline
        }

        // Draw the start and end points
        cv::circle(step_vis, start_pt, 8, cv::Scalar(0, 255, 0), -1); // Green start marker
        cv::circle(step_vis, end_pt, 8, cv::Scalar(255, 0, 0), -1);   // Red end marker
        cv::line(step_vis, start_pt, end_pt, cv::Scalar(255, 255, 255), 1, cv::LINE_AA); // White ideal path

        // Simplified legend with white semi-transparent background
        int legend_x = clean_map.cols - 270; // Position legend in top-right
        int legend_y = 30;
        int legend_width = 250;
        int legend_height = 70;
        int line_height = 25;
        
        // Create semi-transparent background for legend
        cv::Mat overlay = step_vis.clone();
        cv::rectangle(overlay, cv::Rect(legend_x-10, legend_y-25, legend_width, legend_height), 
                    cv::Scalar(255, 255, 255), -1);
        cv::addWeighted(overlay, 0.7, step_vis, 0.3, 0, step_vis); // 70% opacity
        
        // Draw simplified legend text with larger font
        cv::putText(step_vis, "Orange: Actual GPS Path", cv::Point(legend_x, legend_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2);
        legend_y += line_height;
        
        cv::putText(step_vis, "Yellow: Estimated Position Path", cv::Point(legend_x, legend_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        // Create telemetry table visualization
        telemetry_vis = cv::Scalar(240, 240, 240); // Clear previous frame

        // Draw table headers
        int header_y = 40;
        cv::putText(telemetry_vis, "Step", cv::Point(30, header_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::putText(telemetry_vis, "Actual Position", cv::Point(80, header_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::putText(telemetry_vis, "Estimated Position", cv::Point(290, header_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::putText(telemetry_vis, "Conf", cv::Point(500, header_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::putText(telemetry_vis, "Lat Err", cv::Point(560, header_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::putText(telemetry_vis, "Lng Err", cv::Point(630, header_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::putText(telemetry_vis, "Err (m)", cv::Point(700, header_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

        // Draw horizontal divider
        cv::line(telemetry_vis, cv::Point(20, header_y+10), 
                cv::Point(780, header_y+10), cv::Scalar(0, 0, 0), 1);
        
        // Show last 15 entries or fewer if not enough data yet
        int entries_to_show = std::min(15, step+1);
        int start_idx = std::max(0, step+1 - entries_to_show);
        
        for (int i = 0; i < entries_to_show; i++) {
            int idx = start_idx + i;
            int row_y = header_y + 30 + (i * 30);
            
            // Step number
            cv::putText(telemetry_vis, std::to_string(idx), 
                       cv::Point(30, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            
            // Actual position
            std::ostringstream actual_pos;
            actual_pos << std::fixed << std::setprecision(6) 
                      << flight_path[idx].first << ", " << flight_path[idx].second;
            cv::putText(telemetry_vis, actual_pos.str(), 
                       cv::Point(80, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
            
            // Estimated position
            std::ostringstream est_pos;
            est_pos << std::fixed << std::setprecision(6) 
                   << algorithm_positions[idx].first << ", " << algorithm_positions[idx].second;
            cv::putText(telemetry_vis, est_pos.str(), 
                       cv::Point(290, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
            
            // Confidence with color
            double conf = match_confidences[idx];
            cv::Scalar conf_color;
            if (conf < 0.3)
                conf_color = cv::Scalar(0, 0, 255);      // Red
            else if (conf < 0.6)
                conf_color = cv::Scalar(0, 140, 255);    // Orange-ish (more readable)
            else
                conf_color = cv::Scalar(0, 128, 0);      // Dark green (more readable)
                
            std::ostringstream conf_text;
            conf_text << std::fixed << std::setprecision(2) << conf;
            cv::putText(telemetry_vis, conf_text.str(), 
                       cv::Point(500, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1);
            
            // Calculate errors for this position
            double lat_error_m = (flight_path[idx].first - algorithm_positions[idx].first) * meters_per_degree_lat;
            double lng_error_m = (flight_path[idx].second - algorithm_positions[idx].second) * meters_per_degree_lng;
            double total_error_m = std::sqrt(lat_error_m * lat_error_m + lng_error_m * lng_error_m);
            
            // Lat error with color
            cv::Scalar lat_error_color;
            if (std::abs(lat_error_m) < 5)
                lat_error_color = cv::Scalar(0, 128, 0);      // Green
            else if (std::abs(lat_error_m) < 25)
                lat_error_color = cv::Scalar(0, 140, 255);    // Orange
            else
                lat_error_color = cv::Scalar(0, 0, 255);      // Red
                
            std::ostringstream lat_error_text;
            lat_error_text << std::fixed << std::setprecision(1) << lat_error_m;
            cv::putText(telemetry_vis, lat_error_text.str(), 
                       cv::Point(560, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, lat_error_color, 1);
            
            // Lng error with color
            cv::Scalar lng_error_color;
            if (std::abs(lng_error_m) < 5)
                lng_error_color = cv::Scalar(0, 128, 0);      // Green
            else if (std::abs(lng_error_m) < 25)
                lng_error_color = cv::Scalar(0, 140, 255);    // Orange
            else
                lng_error_color = cv::Scalar(0, 0, 255);      // Red
                
            std::ostringstream lng_error_text;
            lng_error_text << std::fixed << std::setprecision(1) << lng_error_m;
            cv::putText(telemetry_vis, lng_error_text.str(), 
                       cv::Point(630, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, lng_error_color, 1);
            
            // Total error with color
            cv::Scalar error_color;
            if (total_error_m < 10)
                error_color = cv::Scalar(0, 128, 0);      // Green
            else if (total_error_m < 50)
                error_color = cv::Scalar(0, 140, 255);    // Orange
            else
                error_color = cv::Scalar(0, 0, 255);      // Red
                
            std::ostringstream error_text;
            error_text << std::fixed << std::setprecision(1) << total_error_m;
            cv::putText(telemetry_vis, error_text.str(), 
                       cv::Point(700, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, error_color, 1);
            
            // Light gray divider between rows
            if (i < entries_to_show - 1) {
                cv::line(telemetry_vis, cv::Point(30, row_y+10), 
                        cv::Point(770, row_y+10), cv::Scalar(200, 200, 200), 1);
            }
        }

        // Zoom in on the path by creating a cropped version of the visualization
        // Add padding around the bounding box (20% of size)
        int width = max_x - min_x;
        int height = max_y - min_y;
        int padding_x = width * 0.2;
        int padding_y = height * 0.2;
        
        // Ensure minimum padding
        padding_x = std::max(padding_x, 50);
        padding_y = std::max(padding_y, 50);
        
        // Create crop region with padding
        cv::Rect zoom_rect(
            std::max(0, min_x - padding_x),
            std::max(0, min_y - padding_y),
            std::min(clean_map.cols - std::max(0, min_x - padding_x), width + 2 * padding_x),
            std::min(clean_map.rows - std::max(0, min_y - padding_y), height + 2 * padding_y)
        );
        
        // Make sure zoom_rect is within map boundaries
        zoom_rect = zoom_rect & cv::Rect(0, 0, clean_map.cols, clean_map.rows);
        
        // Only crop if we have a valid rectangle
        cv::Mat zoomed_vis;
        if (zoom_rect.width > 0 && zoom_rect.height > 0) {
            zoomed_vis = step_vis(zoom_rect).clone();
        } else {
            zoomed_vis = step_vis.clone();
        }

        // Update displays
        cv::imshow("Drone Simulation", zoomed_vis);
        cv::imshow("Drone View", drone_view);
        cv::imshow("Telemetry Data", telemetry_vis);

        // Check if destination reached (within ~10 meters)
        double dist_to_end_lat = (end_lat - flight_path.back().first) * meters_per_degree_lat;
        double dist_to_end_lng = (end_lng - flight_path.back().second) * meters_per_degree_lng;
        double dist_to_end = std::sqrt(dist_to_end_lat * dist_to_end_lat + dist_to_end_lng * dist_to_end_lng);

        if (dist_to_end < 10.0)
        {
            std::cout << "Destination reached!" << std::endl;
            break;
        }

        // Add delay for visualization and check for user exit
        char key = cv::waitKey(100); // 100ms delay
        if (key == 27 || key == 'q')
        { // ESC or 'q' to quit
            break;
        }
    }
    
    // Close the telemetry file
    telemetry_file.close();
    std::cout << "Telemetry data saved to drone_telemetry.csv" << std::endl;

    std::cout << "Simulation completed. Press any key to continue..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// Modified function to support waypoint navigation
void runDroneSimulationWithWaypoints(
    const cv::Mat &clean_map,
    const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash> &reference_crops,
    const std::vector<std::pair<double, double>> &waypoints,
    double meters_per_degree_lat, double meters_per_degree_lng,
    double center_lat, double center_lng,
    int center_x, int center_y,
    double mpp)
{
    if (waypoints.size() < 2) {
        std::cout << "Error: At least 2 waypoints required for navigation" << std::endl;
        return;
    }

    // Initialize simulation parameters
    double speed = 6.0;                                        // m/s
    double drift = 0.0;                                        // Set to zero for accurate path following
    double sim_dt = 1.0;                                       // time step in seconds
    int crop_size = static_cast<int>(std::round(100.0 / mpp)); // 100m view size
    int max_steps = 500;                                       // Increased for longer paths

    // Create drone simulation starting at first waypoint
    double start_lat = waypoints[0].first;
    double start_lng = waypoints[0].second;
    
    // Calculate initial heading to first waypoint
    double initial_heading = 0.0;
    if (waypoints.size() > 1) {
        double delta_lng = waypoints[1].second - start_lng;
        double delta_lat = waypoints[1].first - start_lat;
        initial_heading = std::atan2(delta_lng * meters_per_degree_lng,
                                     delta_lat * meters_per_degree_lat) *
                           180.0 / M_PI;
        if (initial_heading < 0)
            initial_heading += 360.0;
    }

    // Create drone simulation
    DroneSimulation drone(start_lat, start_lng, initial_heading, speed, drift,
                          meters_per_degree_lat, meters_per_degree_lng,
                          clean_map, crop_size);
    
    // Setup visualization windows
    cv::namedWindow("Drone Simulation", cv::WINDOW_NORMAL);
    cv::namedWindow("Drone View", cv::WINDOW_NORMAL);
    cv::namedWindow("Telemetry Data", cv::WINDOW_NORMAL);
    cv::resizeWindow("Drone Simulation", 1200, 800);

    // Extract reference crop coordinates into a vector for easier access
    std::vector<std::pair<double, double>> ref_crop_coords;
    for (const auto &crop_pair : reference_crops) {
        ref_crop_coords.push_back(crop_pair.first);
    }

    // Store algorithm estimated positions and match confidences
    std::vector<std::pair<double, double>> algorithm_positions;
    std::vector<double> match_confidences;
    
    // Create CSV file for telemetry data
    std::ofstream telemetry_file("drone_telemetry_zigzag.csv");
    telemetry_file << "Step,Actual_Lat,Actual_Lng,Estimated_Lat,Estimated_Lng,Match_Confidence,Lat_Error_M,Lng_Error_M,Error_M,Current_Waypoint\n";
    
    // Create a telemetry data visualization matrix
    int telemetry_height = 600;
    int telemetry_width = 800;
    cv::Mat telemetry_vis(telemetry_height, telemetry_width, CV_8UC3, cv::Scalar(240, 240, 240));
    
    // Variables for path bounding box to enable zooming
    int min_x = INT_MAX, min_y = INT_MAX, max_x = 0, max_y = 0;
    
    // Calculate pixel coordinates for all waypoints for drawing
    std::vector<cv::Point> waypoint_pixels;
    for (const auto &waypoint : waypoints) {
        cv::Point pt = drone.latLngToPixel(waypoint.first, waypoint.second, 
                                         center_lat, center_lng, center_x, center_y, mpp);
        waypoint_pixels.push_back(pt);
        
        // Update bounding box
        min_x = std::min(min_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_x = std::max(max_x, pt.x);
        max_y = std::max(max_y, pt.y);
    }

    // Current target waypoint index (start with waypoint 1 since we're already at waypoint 0)
    int current_waypoint = 1;
    
    // Main simulation loop
    for (int step = 0; step < max_steps; step++)
    {
        // Check if we've reached the current waypoint (within ~10 meters)
        if (current_waypoint < waypoints.size()) {
            double target_lat = waypoints[current_waypoint].first;
            double target_lng = waypoints[current_waypoint].second;
            
            double dist_to_waypoint_lat = (target_lat - drone.getPosition().first) * meters_per_degree_lat;
            double dist_to_waypoint_lng = (target_lng - drone.getPosition().second) * meters_per_degree_lng;
            double dist_to_waypoint = std::sqrt(dist_to_waypoint_lat * dist_to_waypoint_lat + 
                                              dist_to_waypoint_lng * dist_to_waypoint_lng);
            
            // If reached current waypoint, set heading to next waypoint
            if (dist_to_waypoint < 10.0) {
                current_waypoint++;
                
                // If there are more waypoints, calculate new heading
                if (current_waypoint < waypoints.size()) {
                    double next_lat = waypoints[current_waypoint].first;
                    double next_lng = waypoints[current_waypoint].second;
                    
                    double delta_lng = next_lng - drone.getPosition().second;
                    double delta_lat = next_lat - drone.getPosition().first;
                    double new_heading = std::atan2(delta_lng * meters_per_degree_lng,
                                                delta_lat * meters_per_degree_lat) *
                                     180.0 / M_PI;
                    if (new_heading < 0)
                        new_heading += 360.0;
                    
                    // Set drone's heading to point to the next waypoint
                    drone.setHeading(new_heading);
                    
                    std::cout << "Reached waypoint " << current_waypoint - 1 
                             << ", heading to waypoint " << current_waypoint << std::endl;
                }
            }
        }

        // Update drone's position based on current heading
        drone.step(sim_dt, 0.0);

        // Update drone's current view (image capture)
        drone.updateView(center_lat, center_lng, center_x, center_y, mpp);
        cv::Mat drone_view = drone.getCurrentView();

        // Use hybrid image matching to estimate position
        std::pair<double, double> last_position = algorithm_positions.empty() ? 
            drone.getPosition() : algorithm_positions.back();
            
        auto [estimated_position, match_confidence, best_match_idx] =
            estimatePositionHybrid(drone_view, ref_crop_coords, reference_crops, last_position);

        // Add temporal smoothing to reduce sudden jumps
        if (!algorithm_positions.empty()) {
            const double smooth_factor = 0.3; // 0-1, higher means more smoothing
            estimated_position.first = estimated_position.first * (1-smooth_factor) + 
                                   algorithm_positions.back().first * smooth_factor;
            estimated_position.second = estimated_position.second * (1-smooth_factor) + 
                                    algorithm_positions.back().second * smooth_factor;
        }

        // Always store the estimated position regardless of match quality
        algorithm_positions.push_back(estimated_position);
        match_confidences.push_back(match_confidence);

        // Write telemetry data to CSV
        double error_lat_m = (drone.getPosition().first - estimated_position.first) * meters_per_degree_lat;
        double error_lng_m = (drone.getPosition().second - estimated_position.second) * meters_per_degree_lng;
        double error_m = std::sqrt(error_lat_m * error_lat_m + error_lng_m * error_lng_m);

        telemetry_file << step << ","
             << std::fixed << std::setprecision(6) << drone.getPosition().first << ","
             << drone.getPosition().second << ","
             << estimated_position.first << ","
             << estimated_position.second << ","
             << std::setprecision(3) << match_confidence << ","
             << std::setprecision(1) << error_lat_m << ","
             << error_lng_m << ","
             << error_m << ","
             << current_waypoint << "\n";

        // Create visualization for this step
        cv::Mat step_vis = clean_map.clone();

        // Get the flight path so far
        const auto &flight_path = drone.getFlightPath();

        // Collect GPS path points (actual drone path)
        std::vector<cv::Point> gps_path_points;
        for (const auto &position : flight_path) {
            cv::Point pt = drone.latLngToPixel(position.first, position.second,
                                             center_lat, center_lng, center_x, center_y, mpp);
            gps_path_points.push_back(pt);
            
            // Update bounding box for zoom
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_x = std::max(max_x, pt.x);
            max_y = std::max(max_y, pt.y);
        }
        
        // Collect algorithm path points (all estimated positions)
        std::vector<cv::Point> algo_path_points;
        for (size_t i = 0; i < algorithm_positions.size(); i++) {
            cv::Point pt = drone.latLngToPixel(algorithm_positions[i].first, algorithm_positions[i].second,
                                             center_lat, center_lng, center_x, center_y, mpp);
            algo_path_points.push_back(pt);
            
            // Update bounding box for zoom
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_x = std::max(max_x, pt.x);
            max_y = std::max(max_y, pt.y);
        }

        // Draw the planned path (connecting all waypoints)
        for (size_t i = 1; i < waypoint_pixels.size(); i++) {
            cv::line(step_vis, waypoint_pixels[i-1], waypoint_pixels[i], 
                   cv::Scalar(255, 255, 255), 1, cv::LINE_AA); // White path
        }
        
        // Draw the GPS-based path (actual path) - orange
        for (size_t i = 1; i < gps_path_points.size(); i++) {
            cv::line(step_vis, gps_path_points[i-1], gps_path_points[i], 
                    cv::Scalar(0, 165, 255), 5);    // Orange line
        }

        // Draw the algorithm-based path (estimated positions) - yellow
        for (size_t i = 1; i < algo_path_points.size(); i++) {
            cv::line(step_vis, algo_path_points[i-1], algo_path_points[i], 
                    cv::Scalar(0, 255, 255), 4);    // Yellow line
        }

        // Draw markers for all waypoints
        for (size_t i = 0; i < waypoint_pixels.size(); i++) {
            cv::Scalar color;
            int radius = 6;
            
            if (i == 0) {
                color = cv::Scalar(0, 255, 0); // Green for start
                radius = 8;
            } else if (i == waypoint_pixels.size() - 1) {
                color = cv::Scalar(255, 0, 0); // Red for end
                radius = 8;
            } else if (i < current_waypoint) {
                color = cv::Scalar(0, 128, 255); // Orange for visited waypoints
            } else if (i == current_waypoint) {
                color = cv::Scalar(255, 0, 255); // Purple for current target waypoint
                radius = 7;
            } else {
                color = cv::Scalar(150, 150, 150); // Gray for future waypoints
            }
            
            cv::circle(step_vis, waypoint_pixels[i], radius, color, -1);
            cv::circle(step_vis, waypoint_pixels[i], radius, cv::Scalar(0, 0, 0), 1); // Black outline
        }
        
        // Draw current drone position
        if (!gps_path_points.empty()) {
            cv::circle(step_vis, gps_path_points.back(), 6, cv::Scalar(0, 165, 255), -1);
            cv::circle(step_vis, gps_path_points.back(), 6, cv::Scalar(0, 0, 0), 1);
        }

        // Draw current algorithm position
        if (!algo_path_points.empty()) {
            cv::circle(step_vis, algo_path_points.back(), 6, cv::Scalar(0, 255, 255), -1);
            cv::circle(step_vis, algo_path_points.back(), 6, cv::Scalar(0, 0, 0), 1);
        }
        
        // Add legend
        int legend_x = clean_map.cols - 270;
        int legend_y = 30;
        int legend_width = 250;
        int legend_height = 100; // Increased height for more items
        int line_height = 25;
        
        // Create semi-transparent background for legend
        cv::Mat overlay = step_vis.clone();
        cv::rectangle(overlay, cv::Rect(legend_x-10, legend_y-25, legend_width, legend_height), 
                    cv::Scalar(255, 255, 255), -1);
        cv::addWeighted(overlay, 0.7, step_vis, 0.3, 0, step_vis);
        
        // Draw legend text
        cv::putText(step_vis, "Orange: Actual GPS Path", cv::Point(legend_x, legend_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2);
        legend_y += line_height;
        
        cv::putText(step_vis, "Yellow: Estimated Path", cv::Point(legend_x, legend_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        legend_y += line_height;
                  
        cv::putText(step_vis, "White: Planned Path", cv::Point(legend_x, legend_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // Create telemetry table visualization
        telemetry_vis = cv::Scalar(240, 240, 240); // Clear previous frame

        // Draw table headers
        int header_y = 40;
        cv::putText(telemetry_vis, "Step", cv::Point(30, header_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::putText(telemetry_vis, "Actual Position", cv::Point(80, header_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::putText(telemetry_vis, "Estimated Position", cv::Point(290, header_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::putText(telemetry_vis, "Conf", cv::Point(500, header_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::putText(telemetry_vis, "Lat Err", cv::Point(560, header_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::putText(telemetry_vis, "Lng Err", cv::Point(630, header_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::putText(telemetry_vis, "Err (m)", cv::Point(700, header_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

        // Draw horizontal divider
        cv::line(telemetry_vis, cv::Point(20, header_y+10), 
                cv::Point(780, header_y+10), cv::Scalar(0, 0, 0), 1);
        
        // Show last 15 entries or fewer if not enough data yet
        int entries_to_show = std::min(15, step+1);
        int start_idx = std::max(0, step+1 - entries_to_show);
        
        for (int i = 0; i < entries_to_show; i++) {
            int idx = start_idx + i;
            int row_y = header_y + 30 + (i * 30);
            
            // Step number
            cv::putText(telemetry_vis, std::to_string(idx), 
                       cv::Point(30, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            
            // Actual position
            std::ostringstream actual_pos;
            actual_pos << std::fixed << std::setprecision(6) 
                      << flight_path[idx].first << ", " << flight_path[idx].second;
            cv::putText(telemetry_vis, actual_pos.str(), 
                       cv::Point(80, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
            
            // Estimated position
            std::ostringstream est_pos;
            est_pos << std::fixed << std::setprecision(6) 
                   << algorithm_positions[idx].first << ", " << algorithm_positions[idx].second;
            cv::putText(telemetry_vis, est_pos.str(), 
                       cv::Point(290, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
            
            // Confidence with color
            double conf = match_confidences[idx];
            cv::Scalar conf_color;
            if (conf < 0.3)
                conf_color = cv::Scalar(0, 0, 255);      // Red
            else if (conf < 0.6)
                conf_color = cv::Scalar(0, 140, 255);    // Orange-ish (more readable)
            else
                conf_color = cv::Scalar(0, 128, 0);      // Dark green (more readable)
                
            std::ostringstream conf_text;
            conf_text << std::fixed << std::setprecision(2) << conf;
            cv::putText(telemetry_vis, conf_text.str(), 
                       cv::Point(500, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1);
            
            // Calculate errors for this position
            double lat_error_m = (flight_path[idx].first - algorithm_positions[idx].first) * meters_per_degree_lat;
            double lng_error_m = (flight_path[idx].second - algorithm_positions[idx].second) * meters_per_degree_lng;
            double total_error_m = std::sqrt(lat_error_m * lat_error_m + lng_error_m * lng_error_m);
            
            // Lat error with color
            cv::Scalar lat_error_color;
            if (std::abs(lat_error_m) < 5)
                lat_error_color = cv::Scalar(0, 128, 0);      // Green
            else if (std::abs(lat_error_m) < 25)
                lat_error_color = cv::Scalar(0, 140, 255);    // Orange
            else
                lat_error_color = cv::Scalar(0, 0, 255);      // Red
                
            std::ostringstream lat_error_text;
            lat_error_text << std::fixed << std::setprecision(1) << lat_error_m;
            cv::putText(telemetry_vis, lat_error_text.str(), 
                       cv::Point(560, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, lat_error_color, 1);
            
            // Lng error with color
            cv::Scalar lng_error_color;
            if (std::abs(lng_error_m) < 5)
                lng_error_color = cv::Scalar(0, 128, 0);      // Green
            else if (std::abs(lng_error_m) < 25)
                lng_error_color = cv::Scalar(0, 140, 255);    // Orange
            else
                lng_error_color = cv::Scalar(0, 0, 255);      // Red
                
            std::ostringstream lng_error_text;
            lng_error_text << std::fixed << std::setprecision(1) << lng_error_m;
            cv::putText(telemetry_vis, lng_error_text.str(), 
                       cv::Point(630, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, lng_error_color, 1);
            
            // Total error with color
            cv::Scalar error_color;
            if (total_error_m < 10)
                error_color = cv::Scalar(0, 128, 0);      // Green
            else if (total_error_m < 50)
                error_color = cv::Scalar(0, 140, 255);    // Orange
            else
                error_color = cv::Scalar(0, 0, 255);      // Red
                
            std::ostringstream error_text;
            error_text << std::fixed << std::setprecision(1) << total_error_m;
            cv::putText(telemetry_vis, error_text.str(), 
                       cv::Point(700, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, error_color, 1);
            
            // Light gray divider between rows
            if (i < entries_to_show - 1) {
                cv::line(telemetry_vis, cv::Point(30, row_y+10), 
                        cv::Point(770, row_y+10), cv::Scalar(200, 200, 200), 1);
            }
        }

        // Zoom in on the path by creating a cropped version of the visualization
        // Add padding around the bounding box (20% of size)
        int width = max_x - min_x;
        int height = max_y - min_y;
        int padding_x = width * 0.2;
        int padding_y = height * 0.2;
        
        // Ensure minimum padding
        padding_x = std::max(padding_x, 50);
        padding_y = std::max(padding_y, 50);
        
        // Create crop region with padding
        cv::Rect zoom_rect(
            std::max(0, min_x - padding_x),
            std::max(0, min_y - padding_y),
            std::min(clean_map.cols - std::max(0, min_x - padding_x), width + 2 * padding_x),
            std::min(clean_map.rows - std::max(0, min_y - padding_y), height + 2 * padding_y)
        );
        
        // Make sure zoom_rect is within map boundaries
        zoom_rect = zoom_rect & cv::Rect(0, 0, clean_map.cols, clean_map.rows);
        
        // Only crop if we have a valid rectangle
        cv::Mat zoomed_vis;
        if (zoom_rect.width > 0 && zoom_rect.height > 0) {
            zoomed_vis = step_vis(zoom_rect).clone();
        } else {
            zoomed_vis = step_vis.clone();
        }

        // Update displays
        cv::imshow("Drone Simulation", zoomed_vis);
        cv::imshow("Drone View", drone_view);
        cv::imshow("Telemetry Data", telemetry_vis);

        // Check if final destination reached
        if (current_waypoint >= waypoints.size()) {
            std::cout << "All waypoints reached! Simulation complete." << std::endl;
            break;
        }

        // Add delay and check for user exit
        char key = cv::waitKey(100);
        if (key == 27 || key == 'q') {
            break;
        }
    }
    
    telemetry_file.close();
    std::cout << "Telemetry data saved to drone_telemetry.csv" << std::endl;
    std::cout << "Simulation completed. Press any key to continue..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}


std::pair<double, double> getSmoothedPositionEstimate(
    const cv::Mat &drone_view,
    const std::vector<std::pair<double, double>> &ref_crop_coords,
    const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash> &reference_crops,
    const std::pair<double, double> &last_position)
{
    // Get all matches with scores
    std::vector<std::tuple<int, double, std::pair<double, double>>> all_matches;
    
    // Calculate match score for each reference crop
    for (size_t idx = 0; idx < ref_crop_coords.size(); idx++) {
        const auto &coords = ref_crop_coords[idx];
        const auto &ref_crop = reference_crops.at(coords);
        
        // Ensure comparable sizes
        cv::Mat comparison_image;
        if (drone_view.size() != ref_crop.size()) {
            cv::resize(ref_crop, comparison_image, drone_view.size());
        } else {
            comparison_image = ref_crop;
        }
        
        // Calculate match score
        cv::Mat result;
        cv::matchTemplate(comparison_image, drone_view, result, cv::TM_CCOEFF_NORMED);
        
        double score;
        cv::minMaxLoc(result, nullptr, &score);
        all_matches.push_back(std::make_tuple(idx, score, coords));
    }
    
    // Sort by score in descending order
    std::sort(all_matches.begin(), all_matches.end(),
        [](const auto &a, const auto &b) {
            return std::get<1>(a) > std::get<1>(b);
        });
    
    // Use weighted average of top 3 matches (or fewer if not enough matches)
    int matches_to_use = std::min(3, static_cast<int>(all_matches.size()));
    if (matches_to_use == 0) {
        return last_position; // No matches, return last known position
    }
    
    double total_weight = 0.0;
    double weighted_lat = 0.0;
    double weighted_lng = 0.0;
    
    for (int i = 0; i < matches_to_use; i++) {
        const auto &match = all_matches[i];
        double score = std::get<1>(match);
        const auto &coords = std::get<2>(match);
        
        // Apply score^2 weighting to emphasize better matches
        double weight = score * score;
        total_weight += weight;
        weighted_lat += coords.first * weight;
        weighted_lng += coords.second * weight;
    }
    
    if (total_weight > 0) {
        return std::make_pair(weighted_lat / total_weight, weighted_lng / total_weight);
    } else {
        return last_position; // Fallback
    }
}

// Add to estimatePositionFromImage or create a new feature-based version
std::pair<double, double> estimatePositionWithFeatures(
    const cv::Mat &drone_view,
    const std::vector<std::pair<double, double>> &ref_crop_coords,
    const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash> &reference_crops) 
{
    // Create ORB detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);  // 500 keypoints
    
    // Extract keypoints and descriptors from drone view
    std::vector<cv::KeyPoint> keypoints_drone;
    cv::Mat descriptors_drone;
    orb->detectAndCompute(drone_view, cv::noArray(), keypoints_drone, descriptors_drone);
    
    if (keypoints_drone.size() < 10) {
        // Not enough features in drone view
        return {0.0, 0.0};
    }
    
    // Match against each reference crop
    std::vector<std::pair<int, float>> match_scores;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    
    for (size_t idx = 0; idx < ref_crop_coords.size(); idx++) {
        const auto &coords = ref_crop_coords[idx];
        const cv::Mat &ref_crop = reference_crops.at(coords);
        
        // Extract features from reference crop
        std::vector<cv::KeyPoint> keypoints_ref;
        cv::Mat descriptors_ref;
        orb->detectAndCompute(ref_crop, cv::noArray(), keypoints_ref, descriptors_ref);
        
        if (keypoints_ref.size() < 10) continue;
        
        // Match descriptors
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(descriptors_drone, descriptors_ref, knn_matches, 2);
        
        // Filter good matches using ratio test
        std::vector<cv::DMatch> good_matches;
        for (const auto &match_pair : knn_matches) {
            if (match_pair.size() >= 2) {
                if (match_pair[0].distance < 0.75f * match_pair[1].distance) {
                    good_matches.push_back(match_pair[0]);
                }
            }
        }
        
        // Calculate score based on number and quality of matches
        float match_score = 0.0f;
        if (!good_matches.empty()) {
            // Score combines number of good matches and their quality
            float avg_distance = 0.0f;
            for (const auto &m : good_matches) {
                avg_distance += m.distance;
            }
            avg_distance /= good_matches.size();
            
            // Higher score for more matches and lower distance
            match_score = good_matches.size() / (avg_distance + 1.0f);
        }
        
        match_scores.push_back({idx, match_score});
    }
    
    // Sort by score
    std::sort(match_scores.begin(), match_scores.end(),
             [](const auto &a, const auto &b) { return a.second > b.second; });
    
    // Return coordinates of best match or weighted average
    if (match_scores.empty()) {
        return {0.0, 0.0};
    }
    
    // Get top match
    int best_idx = match_scores[0].first;
    return ref_crop_coords[best_idx];
}

// Hybrid position estimation using both template and feature matching
std::tuple<std::pair<double, double>, double, int> estimatePositionHybrid(
    const cv::Mat &drone_view,
    const std::vector<std::pair<double, double>> &ref_crop_coords,
    const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash> &reference_crops,
    const std::pair<double, double> &last_position)
{
    // Get position estimate using template matching
    auto [template_position, template_confidence, template_idx] =
        estimatePositionFromImage(drone_view, ref_crop_coords, reference_crops);
    
    // Get position estimate using feature matching
    std::pair<double, double> feature_position =
        estimatePositionWithFeatures(drone_view, ref_crop_coords, reference_crops);
    
    // Find index and confidence for feature match
    int feature_idx = -1;
    for (size_t i = 0; i < ref_crop_coords.size(); i++) {
        if (ref_crop_coords[i] == feature_position) {
            feature_idx = i;
            break;
        }
    }
    
    // Decide which method to use based on template matching confidence
    if (template_confidence > 0.8) {
        // If template matching is very confident, use it
        return {template_position, template_confidence, template_idx};
    }
    else if (feature_idx != -1) {
        // Otherwise use feature matching result
        // Assign a confidence value (you may want to adjust this)
        double feature_confidence = 0.7;
        return {feature_position, feature_confidence, feature_idx};
    }
    else {
        // If both methods fail, use last known position with low confidence
        return {last_position, 0.3, -1};
    }
}
