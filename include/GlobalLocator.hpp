#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include <string>
#include <pthread.h>
#include "DroneSimulation.hpp"

struct InitializationData {
    bool success;
    int frame_index; // The frame number where we finally got a lock
    std::pair<double, double> coordinates;
};

class GlobalLocator {
public:

    /**
     * @brief Freezes execution and uses POSIX threads to search all reference crops 
     * to find the drone's starting location.
     * * @param video_path Path to the video file
     * @param crops The full database of reference crops (e.g., 2598 crops)
     * @param num_threads Number of POSIX threads to spawn
     * @return The exact starting coordinate {lat, lng}
     */
    static InitializationData findStartingPosition(
        const std::string& video_path,
        const std::vector<ReferenceCrop>& crops,
        int num_threads = 8,
        int frame_skip = 10
    );
};