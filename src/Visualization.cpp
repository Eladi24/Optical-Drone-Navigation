#include "Visualization.hpp"
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>

cv::Mat createSimulationVisualization(const SimulationState& state) {
    cv::Mat step_vis = state.clean_map.clone();

    // Draw planned path
    if (state.is_waypoint_mode) {
        for (size_t i = 1; i < state.waypoint_pixels.size(); i++) {
            cv::line(step_vis, state.waypoint_pixels[i-1], state.waypoint_pixels[i], 
                   cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    } else {
        cv::line(step_vis, state.waypoint_pixels[0], state.waypoint_pixels.back(), 
               cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    
    // Draw GPS path (orange)
    for (size_t i = 1; i < state.gps_path_points.size(); i++) {
        cv::line(step_vis, state.gps_path_points[i-1], state.gps_path_points[i], 
                cv::Scalar(0, 165, 255), 5);
    }

    // Draw estimated path (yellow)
    for (size_t i = 1; i < state.estimated_path_points.size(); i++) {
        cv::line(step_vis, state.estimated_path_points[i-1], state.estimated_path_points[i], 
                cv::Scalar(0, 255, 255), 4);
    }

    // Draw waypoint markers
    for (size_t i = 0; i < state.waypoint_pixels.size(); i++) {
        cv::Scalar color;
        int radius = 6;
        
        if (i == 0) {
            color = cv::Scalar(0, 255, 0); // Green for start
            radius = 8;
        } else if (i == state.waypoint_pixels.size() - 1) {
            color = cv::Scalar(255, 0, 0); // Red for end
            radius = 8;
        } else if (state.is_waypoint_mode) {
            if (i < state.current_waypoint) {
                color = cv::Scalar(0, 128, 255); // Orange for visited
            } else if (i == state.current_waypoint) {
                color = cv::Scalar(255, 0, 255); // Purple for current target
                radius = 7;
            } else {
                color = cv::Scalar(150, 150, 150); // Gray for future
            }
        } else {
            continue;
        }
        
        cv::circle(step_vis, state.waypoint_pixels[i], radius, color, -1);
        cv::circle(step_vis, state.waypoint_pixels[i], radius, cv::Scalar(0, 0, 0), 1);
    }
    
    // Draw current positions
    if (!state.gps_path_points.empty()) {
        cv::circle(step_vis, state.gps_path_points.back(), 6, cv::Scalar(0, 165, 255), -1);
        cv::circle(step_vis, state.gps_path_points.back(), 6, cv::Scalar(0, 0, 0), 1);
    }

    if (!state.estimated_path_points.empty()) {
        cv::circle(step_vis, state.estimated_path_points.back(), 6, cv::Scalar(0, 255, 255), -1);
        cv::circle(step_vis, state.estimated_path_points.back(), 6, cv::Scalar(0, 0, 0), 1);
    }
    
    // Draw legend
    int legend_x = state.clean_map.cols - 270;
    int legend_y = 30;
    int legend_width = 250;
    int legend_height = state.is_waypoint_mode ? 130 : 70;
    
    cv::Mat overlay = step_vis.clone();
    cv::rectangle(overlay, cv::Rect(legend_x-10, legend_y-25, legend_width, legend_height), 
                cv::Scalar(255, 255, 255), -1);
    cv::addWeighted(overlay, 0.7, step_vis, 0.3, 0, step_vis);
    
    if (state.is_waypoint_mode) {
        cv::putText(step_vis, "Algorithm: " + state.algorithm_name, 
                   cv::Point(legend_x, legend_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 200), 2);
        legend_y += 25;
    }
    
    cv::putText(step_vis, "Orange: Actual GPS Path", cv::Point(legend_x, legend_y), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2);
    legend_y += 25;
    
    cv::putText(step_vis, state.is_waypoint_mode ? "Yellow: Estimated Path" : "Yellow: Estimated Position Path", 
               cv::Point(legend_x, legend_y), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

    return step_vis;
}

cv::Mat createTelemetryVisualization(const TelemetryData& data, int current_step) {
    int telemetry_height = 600;
    int telemetry_width = 800;
    cv::Mat telemetry_vis(telemetry_height, telemetry_width, CV_8UC3, cv::Scalar(240, 240, 240));

    int header_y = data.is_waypoint_mode ? 60 : 40;
    if (data.is_waypoint_mode) {
        cv::putText(telemetry_vis, "Algorithm: " + data.algorithm_name, 
                   cv::Point(20, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 200), 2);
    }

    // Draw headers
    cv::putText(telemetry_vis, "Step", cv::Point(30, header_y), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    cv::putText(telemetry_vis, "Actual Position", cv::Point(80, header_y), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    cv::putText(telemetry_vis, "Estimated Position", cv::Point(290, header_y), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    cv::putText(telemetry_vis, "Conf", cv::Point(500, header_y), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    cv::putText(telemetry_vis, "Err (m)", cv::Point(700, header_y), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    cv::line(telemetry_vis, cv::Point(20, header_y+10), 
            cv::Point(780, header_y+10), cv::Scalar(0, 0, 0), 1);
    
    // Show last 15 entries
    int entries_to_show = std::min(15, static_cast<int>(data.entries.size()));
    int start_idx = std::max(0, static_cast<int>(data.entries.size()) - entries_to_show);
    
    for (int i = 0; i < entries_to_show; i++) {
        const auto& entry = data.entries[start_idx + i];
        int row_y = header_y + 30 + (i * 30);
        
        cv::putText(telemetry_vis, std::to_string(entry.step), 
                   cv::Point(30, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        
        std::ostringstream actual_pos;
        actual_pos << std::fixed << std::setprecision(6) 
                  << entry.actual_position.first << ", " << entry.actual_position.second;
        cv::putText(telemetry_vis, actual_pos.str(), 
                   cv::Point(80, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        
        std::ostringstream est_pos;
        est_pos << std::fixed << std::setprecision(6) 
               << entry.estimated_position.first << ", " << entry.estimated_position.second;
        cv::putText(telemetry_vis, est_pos.str(), 
                   cv::Point(290, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        
        cv::Scalar conf_color;
        if (entry.confidence < 0.3) conf_color = cv::Scalar(0, 0, 255);
        else if (entry.confidence < 0.6) conf_color = cv::Scalar(0, 140, 255);
        else conf_color = cv::Scalar(0, 128, 0);
            
        std::ostringstream conf_text;
        conf_text << std::fixed << std::setprecision(2) << entry.confidence;
        cv::putText(telemetry_vis, conf_text.str(), 
                   cv::Point(500, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1);
        
        cv::Scalar error_color;
        if (entry.total_error_m < 10) error_color = cv::Scalar(0, 128, 0);
        else if (entry.total_error_m < 50) error_color = cv::Scalar(0, 140, 255);
        else error_color = cv::Scalar(0, 0, 255);
            
        std::ostringstream error_text;
        error_text << std::fixed << std::setprecision(1) << entry.total_error_m;
        cv::putText(telemetry_vis, error_text.str(), 
                   cv::Point(700, row_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, error_color, 1);
    }

    return telemetry_vis;
}

void updateAllDisplays(
    const cv::Mat& simulation_vis,
    const cv::Mat& drone_view,
    const cv::Mat& telemetry_vis) 
{
    cv::imshow("Drone Simulation", simulation_vis);
    cv::imshow("Drone View", drone_view);
    cv::imshow("Telemetry Data", telemetry_vis);
}

cv::Mat createErrorGraph(
    const std::vector<std::pair<double, double>> &actual_positions,
    const std::vector<std::pair<double, double>> &estimated_positions,
    double meters_per_degree_lat,
    double meters_per_degree_lng,
    const std::string &algorithm_name)
{
    // Add debug output
    std::cout << "Creating error graph..." << std::endl;
    std::cout << "Actual positions: " << actual_positions.size() << std::endl;
    std::cout << "Estimated positions: " << estimated_positions.size() << std::endl;
    
    // Graph dimensions - INCREASED SIZE for better spacing
    int graph_width = 1100;
    int graph_height = 700;
    int margin_left = 80;
    int margin_right = 220;  // Increased to fit statistics box
    int margin_top = 80;     // Increased for title
    int margin_bottom = 160; // Increased for legend
    int plot_width = graph_width - margin_left - margin_right;
    int plot_height = graph_height - margin_top - margin_bottom;
    
    cv::Mat graph(graph_height, graph_width, CV_8UC3, cv::Scalar(240, 240, 240));
    
    // Calculate errors for each position
    std::vector<double> errors;
    double max_error = 0.0;

    // 🆕 ADD SIZE VALIDATION
    if (actual_positions.empty() || estimated_positions.empty()) {
        std::cerr << "Error: No position data for graph generation" << std::endl;
        return graph;
    }

    if (actual_positions.size() != estimated_positions.size()) {
        std::cerr << "Warning: Mismatch in position array sizes: " 
                  << actual_positions.size() << " vs " << estimated_positions.size() << std::endl;
        // Use the smaller size
        size_t min_size = std::min(actual_positions.size(), estimated_positions.size());
    }

    for (size_t i = 0; i < actual_positions.size(); i++) {
        double lat_error = (actual_positions[i].first - estimated_positions[i].first) * meters_per_degree_lat;
        double lng_error = (actual_positions[i].second - estimated_positions[i].second) * meters_per_degree_lng;
        double error = std::sqrt(lat_error * lat_error + lng_error * lng_error);
        
        // 🆕 VALIDATE ERROR VALUE
        if (std::isnan(error) || std::isinf(error)) {
            std::cerr << "Invalid error at step " << i << ": " << error << std::endl;
            error = 0.0;
        }
        
        errors.push_back(error);
        max_error = std::max(max_error, error);
    }
    
    // Round up max_error to nearest 10
    max_error = std::ceil(max_error / 10.0) * 10.0;
    if (max_error < 10.0) max_error = 10.0;
    
    // Draw graph border and background
    cv::rectangle(graph, 
                 cv::Point(margin_left, margin_top),
                 cv::Point(graph_width - margin_right, graph_height - margin_bottom),
                 cv::Scalar(255, 255, 255), -1);
    cv::rectangle(graph, 
                 cv::Point(margin_left, margin_top),
                 cv::Point(graph_width - margin_right, graph_height - margin_bottom),
                 cv::Scalar(0, 0, 0), 2);
    
    // Draw grid lines and Y-axis labels
    int num_y_divisions = 5;
    for (int i = 0; i <= num_y_divisions; i++) {
        int y = margin_top + (plot_height * i) / num_y_divisions;
        
        // Grid line
        cv::line(graph,
                cv::Point(margin_left, y),
                cv::Point(graph_width - margin_right, y),
                cv::Scalar(220, 220, 220), 1);
        
        // Y-axis label
        double error_value = max_error * (1.0 - (double)i / num_y_divisions);
        std::ostringstream label;
        label << std::fixed << std::setprecision(1) << error_value << "m";
        cv::putText(graph, label.str(),
                   cv::Point(10, y + 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    
    // Draw X-axis labels (step numbers)
    int num_x_divisions = std::min(10, (int)errors.size());
    for (int i = 0; i <= num_x_divisions; i++) {
        int step = (errors.size() - 1) * i / num_x_divisions;
        int x = margin_left + (plot_width * i) / num_x_divisions;
        
        // X-axis label
        cv::putText(graph, std::to_string(step),
                   cv::Point(x - 15, graph_height - margin_bottom + 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    
    // Draw error line
    for (size_t i = 1; i < errors.size(); i++) {
        // Calculate x positions
        int x1 = margin_left + (plot_width * (i - 1)) / (errors.size() - 1);
        int x2 = margin_left + (plot_width * i) / (errors.size() - 1);
        
        // Calculate y positions with bounds checking
        double normalized_y1 = std::min(1.0, std::max(0.0, errors[i - 1] / max_error));
        double normalized_y2 = std::min(1.0, std::max(0.0, errors[i] / max_error));
        
        int y1 = margin_top + plot_height - static_cast<int>(plot_height * normalized_y1);
        int y2 = margin_top + plot_height - static_cast<int>(plot_height * normalized_y2);
        
        // Ensure y coordinates are within bounds
        y1 = std::max(margin_top, std::min(graph_height - margin_bottom, y1));
        y2 = std::max(margin_top, std::min(graph_height - margin_bottom, y2));
        
        // Color based on error magnitude
        cv::Scalar line_color;
        double avg_error = (errors[i - 1] + errors[i]) / 2.0;
        if (avg_error < 10.0)
            line_color = cv::Scalar(0, 200, 0);      // Green
        else if (avg_error < 30.0)
            line_color = cv::Scalar(0, 165, 255);    // Orange
        else
            line_color = cv::Scalar(0, 0, 255);      // Red
        
        cv::line(graph, cv::Point(x1, y1), cv::Point(x2, y2), line_color, 2);
    }
    
    // Calculate statistics
    double mean_error = 0.0;
    for (double error : errors) {
        mean_error += error;
    }
    mean_error /= errors.size();
    
    double max_observed = *std::max_element(errors.begin(), errors.end());
    double min_observed = *std::min_element(errors.begin(), errors.end());
    
    // Draw title - centered at top
    cv::putText(graph, "Position Error Over Time - " + algorithm_name,
               cv::Point(graph_width / 2 - 250, 40),
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    
    // Draw axis labels
    cv::putText(graph, "Step",
               cv::Point(graph_width / 2 - 20, graph_height - margin_bottom + 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    
    // Draw legend for color coding
    int legend_x = margin_left + 20;
    int legend_y = graph_height - margin_bottom + 80;
    
    // Green line
    cv::line(graph, 
            cv::Point(legend_x, legend_y),
            cv::Point(legend_x + 40, legend_y),
            cv::Scalar(0, 200, 0), 3);
    cv::putText(graph, "< 10m (Good)",
               cv::Point(legend_x + 50, legend_y + 3),
               cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    
    // Orange line
    cv::line(graph, 
            cv::Point(legend_x, legend_y + 23),
            cv::Point(legend_x + 40, legend_y + 23),
            cv::Scalar(0, 165, 255), 3);
    cv::putText(graph, "10-30m (Fair)",
               cv::Point(legend_x + 50, legend_y + 26),
               cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    
    // Red line
    cv::line(graph, 
            cv::Point(legend_x, legend_y + 46),
            cv::Point(legend_x + 40, legend_y + 46),
            cv::Scalar(0, 0, 255), 3);
    cv::putText(graph, "> 30m (Poor)",
               cv::Point(legend_x + 50, legend_y + 49),
               cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    
    // Draw statistics box on the right side
    int stats_x = graph_width - margin_right + 10;
    int stats_y = margin_top + 20;
    
    cv::putText(graph, "Statistics:",
               cv::Point(stats_x, stats_y),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    std::ostringstream mean_text;
    mean_text << "Mean: " << std::fixed << std::setprecision(1) << mean_error << "m";
    cv::putText(graph, mean_text.str(),
               cv::Point(stats_x, stats_y + 25),
               cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    
    std::ostringstream max_text;
    max_text << "Max: " << std::fixed << std::setprecision(1) << max_observed << "m";
    cv::putText(graph, max_text.str(),
               cv::Point(stats_x, stats_y + 50),
               cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    
    std::ostringstream min_text;
    min_text << "Min: " << std::fixed << std::setprecision(1) << min_observed << "m";
    cv::putText(graph, min_text.str(),
               cv::Point(stats_x, stats_y + 75),
               cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    
    return graph;
}

cv::Mat createErrorGraph(
    const std::vector<double>& errors,
    const std::string &algorithm_name)
{
    std::cout << "Creating error graph..." << std::endl;
    std::cout << "Error values: " << errors.size() << std::endl;
    
    // Graph dimensions
    int graph_width = 1100;
    int graph_height = 700;
    int margin_left = 80;
    int margin_right = 220;
    int margin_top = 80;
    int margin_bottom = 160;
    int plot_width = graph_width - margin_left - margin_right;
    int plot_height = graph_height - margin_top - margin_bottom;
    
    cv::Mat graph(graph_height, graph_width, CV_8UC3, cv::Scalar(240, 240, 240));
    
    // Validate input
    if (errors.empty()) {
        std::cerr << "Error: No error data for graph generation" << std::endl;
        cv::putText(graph, "No data available", cv::Point(400, 350),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
        return graph;
    }

    // Find max error for scaling
    double max_error = 0.0;
    for (double error : errors) {
        if (std::isnan(error) || std::isinf(error)) {
            std::cerr << "Warning: Invalid error value detected: " << error << std::endl;
            continue;
        }
        max_error = std::max(max_error, error);
    }
    
    // Round up max_error to nearest 10
    max_error = std::ceil(max_error / 10.0) * 10.0;
    if (max_error < 10.0) max_error = 10.0;
    
    std::cout << "Graph max error scale: " << max_error << "m" << std::endl;
    
    // Draw graph border and background
    cv::rectangle(graph, 
                 cv::Point(margin_left, margin_top),
                 cv::Point(graph_width - margin_right, graph_height - margin_bottom),
                 cv::Scalar(255, 255, 255), -1);
    cv::rectangle(graph, 
                 cv::Point(margin_left, margin_top),
                 cv::Point(graph_width - margin_right, graph_height - margin_bottom),
                 cv::Scalar(0, 0, 0), 2);
    
    // Draw grid lines and Y-axis labels
    int num_y_divisions = 5;
    for (int i = 0; i <= num_y_divisions; i++) {
        int y = margin_top + (plot_height * i) / num_y_divisions;
        
        cv::line(graph,
                cv::Point(margin_left, y),
                cv::Point(graph_width - margin_right, y),
                cv::Scalar(220, 220, 220), 1);
        
        double error_value = max_error * (1.0 - (double)i / num_y_divisions);
        std::ostringstream label;
        label << std::fixed << std::setprecision(1) << error_value << "m";
        cv::putText(graph, label.str(),
                   cv::Point(10, y + 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    
    // Draw X-axis labels
    int num_x_divisions = std::min(10, (int)errors.size());
    for (int i = 0; i <= num_x_divisions; i++) {
        int step = (errors.size() - 1) * i / num_x_divisions;
        int x = margin_left + (plot_width * i) / num_x_divisions;
        
        cv::putText(graph, std::to_string(step),
                   cv::Point(x - 15, graph_height - margin_bottom + 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    
    // Draw error line
    for (size_t i = 1; i < errors.size(); i++) {
        int x1 = margin_left + (plot_width * (i - 1)) / (errors.size() - 1);
        int x2 = margin_left + (plot_width * i) / (errors.size() - 1);
        
        // Clamp errors to valid range
        double clamped_error1 = std::min(max_error, std::max(0.0, errors[i - 1]));
        double clamped_error2 = std::min(max_error, std::max(0.0, errors[i]));
        
        double normalized_y1 = clamped_error1 / max_error;
        double normalized_y2 = clamped_error2 / max_error;
        
        int y1 = margin_top + plot_height - static_cast<int>(plot_height * normalized_y1);
        int y2 = margin_top + plot_height - static_cast<int>(plot_height * normalized_y2);
        
        // Color based on error magnitude
        cv::Scalar line_color;
        double avg_error = (errors[i - 1] + errors[i]) / 2.0;
        if (avg_error < 10.0)
            line_color = cv::Scalar(0, 200, 0);
        else if (avg_error < 30.0)
            line_color = cv::Scalar(0, 165, 255);
        else
            line_color = cv::Scalar(0, 0, 255);
        
        cv::line(graph, cv::Point(x1, y1), cv::Point(x2, y2), line_color, 2);
    }
    
    // Calculate statistics
    double mean_error = 0.0;
    for (double error : errors) {
        mean_error += error;
    }
    mean_error /= errors.size();
    
    double max_observed = *std::max_element(errors.begin(), errors.end());
    double min_observed = *std::min_element(errors.begin(), errors.end());
    
    std::cout << "Graph statistics - Mean: " << mean_error 
              << "m, Max: " << max_observed << "m, Min: " << min_observed << "m" << std::endl;
    
    // Draw title
    cv::putText(graph, "Position Error Over Time - " + algorithm_name,
               cv::Point(graph_width / 2 - 250, 40),
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    
    // Draw axis labels
    cv::putText(graph, "Step",
               cv::Point(graph_width / 2 - 20, graph_height - margin_bottom + 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    
    // Draw legend
    int legend_x = margin_left + 20;
    int legend_y = graph_height - margin_bottom + 80;
    
    cv::line(graph, cv::Point(legend_x, legend_y), cv::Point(legend_x + 40, legend_y),
            cv::Scalar(0, 200, 0), 3);
    cv::putText(graph, "< 10m (Good)", cv::Point(legend_x + 50, legend_y + 3),
               cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    
    cv::line(graph, cv::Point(legend_x, legend_y + 23), cv::Point(legend_x + 40, legend_y + 23),
            cv::Scalar(0, 165, 255), 3);
    cv::putText(graph, "10-30m (Fair)", cv::Point(legend_x + 50, legend_y + 26),
               cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    
    cv::line(graph, cv::Point(legend_x, legend_y + 46), cv::Point(legend_x + 40, legend_y + 46),
            cv::Scalar(0, 0, 255), 3);
    cv::putText(graph, "> 30m (Poor)", cv::Point(legend_x + 50, legend_y + 49),
               cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    
    // Draw statistics box
    int stats_x = graph_width - margin_right + 10;
    int stats_y = margin_top + 20;
    
    cv::putText(graph, "Statistics:", cv::Point(stats_x, stats_y),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    std::ostringstream mean_text, max_text, min_text;
    mean_text << "Mean: " << std::fixed << std::setprecision(1) << mean_error << "m";
    max_text << "Max: " << std::fixed << std::setprecision(1) << max_observed << "m";
    min_text << "Min: " << std::fixed << std::setprecision(1) << min_observed << "m";
    
    cv::putText(graph, mean_text.str(), cv::Point(stats_x, stats_y + 25),
               cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    cv::putText(graph, max_text.str(), cv::Point(stats_x, stats_y + 50),
               cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    cv::putText(graph, min_text.str(), cv::Point(stats_x, stats_y + 75),
               cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    
    return graph;
}