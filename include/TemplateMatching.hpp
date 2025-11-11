#pragma once

#include <unordered_map>
#include <utility>
#include "DroneSimulation.hpp"
#include "PositionEstimation.hpp"

void matchCropsOnMap(
    const cv::Mat& clean_map,
    const std::vector<ReferenceCrop> &crops);

void runDroneSimulation(
    const cv::Mat &clean_map,
    const std::vector<ReferenceCrop> &reference_crops,
    const std::vector<std::pair<double, double>> &waypoints,
    double meters_per_degree_lat, double meters_per_degree_lng,
    double center_lat, double center_lng,
    int center_x, int center_y,
    double mpp,
    PositionAlgorithm algorithm);

