#pragma once

#include <unordered_map>
#include <utility>
#include "DroneSimulation.hpp"
#include "PositionEstimation.hpp"
// enum class PositionAlgorithm {
//     TEMPLATE,    // Template matching only
//     ORB,         // ORB feature matching
//     DNN,         // Deep neural network features
//     HYBRID,      // Combination of methods (your current approach)
//     SMOOTHED     // Smoothed weighted average method
// };


void matchCropsOnMap(
    const cv::Mat& clean_map,
    const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash>& crops);

void runDroneSimulation(
    const cv::Mat &clean_map,
    const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash> &reference_crops,
    const std::vector<std::pair<double, double>> &waypoints,
    double meters_per_degree_lat, double meters_per_degree_lng,
    double center_lat, double center_lng,
    int center_x, int center_y,
    double mpp,
    PositionAlgorithm algorithm);

