#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "DroneSimulation.hpp"
#include "PositionEstimation.hpp"
#include "KalmanFilter.hpp"

/**
 * Generate a grid of reference crops covering the entire map area
 * This is used when we don't know the flight path in advance (real video)
 */
std::vector<ReferenceCrop> generateReferenceCropsGrid(
    const cv::Mat& map,
    double center_lat, double center_lng,
    int center_x, int center_y,
    double mpp,
    double meters_per_degree_lat,
    double meters_per_degree_lng,
    int grid_spacing_meters = 100,
    int crop_size_meters = 100
);

/**
 * Process a video file and estimate the drone's position throughout the flight
 * 
 * @param video_path Path to the video file
 * @param reference_map The satellite/aerial reference map
 * @param reference_crops Grid of reference crops covering the map area
 * @param center_lat Center latitude of the reference map
 * @param center_lng Center longitude of the reference map
 * @param center_x Center x pixel coordinate of the map
 * @param center_y Center y pixel coordinate of the map
 * @param mpp Meters per pixel at the map's zoom level
 * @param meters_per_degree_lat Meters per degree latitude at this location
 * @param meters_per_degree_lng Meters per degree longitude at this location
 * @param algorithm Position estimation algorithm to use
 * @param location_name Name of the location (e.g., "haifa")
 * @param video_name Name/identifier for this video (e.g., "sample1")
 * @param frame_skip Process every Nth frame (default: 1 = every frame)
 * @param initial_position Optional initial position estimate {lat, lng}
 */
void processVideoNavigation(
    const std::string& video_path,
    const cv::Mat& reference_map,
    const std::vector<ReferenceCrop>& reference_crops,
    double center_lat, double center_lng,
    int center_x, int center_y,
    double mpp,
    double meters_per_degree_lat,
    double meters_per_degree_lng,
    PositionAlgorithm algorithm,
    const std::string& location_name,
    const std::string& video_name,
    int frame_skip = 1,
    std::pair<double, double> initial_position = {0.0, 0.0},
    int start_frame_idx = 0
);
