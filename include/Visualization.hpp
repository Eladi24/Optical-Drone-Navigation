#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Data structures for visualization
struct SimulationState {
    cv::Mat clean_map;
    std::vector<cv::Point> waypoint_pixels;
    std::vector<cv::Point> gps_path_points;
    std::vector<cv::Point> estimated_path_points;
    int current_waypoint;
    int total_waypoints;
    std::string algorithm_name;
    bool is_waypoint_mode;
};

struct TelemetryEntry {
    int step;
    std::pair<double, double> actual_position;
    std::pair<double, double> estimated_position;
    double confidence;
    double lat_error_m;
    double lng_error_m;
    double total_error_m;
};

struct TelemetryData {
    std::vector<TelemetryEntry> entries;
    std::string algorithm_name;
    bool is_waypoint_mode;
};

// Visualization functions
cv::Mat createSimulationVisualization(const SimulationState& state);

cv::Mat createTelemetryVisualization(const TelemetryData& data, int current_step);

cv::Mat createErrorGraph(
    const std::vector<double>& error_values,  // Pre-calculated errors
    const std::string &algorithm_name);

// Helper to update all visualization windows at once
void updateAllDisplays(
    const cv::Mat& simulation_vis,
    const cv::Mat& drone_view,
    const cv::Mat& telemetry_vis);