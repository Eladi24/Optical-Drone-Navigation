#pragma once

#include <cmath>

// Shared pinhole-camera scale math (altitude <-> footprint <-> ground-sample-distance).
//
// Previously this 70-degree HFOV assumption existed as three separate, disconnected
// literals (VideoPreprocessor.hpp's estimateAltitudeFromReference default parameter,
// the explicit 70.0 passed at its one call site in VideoPreprocessor.cpp, and an
// independent constexpr in Main.cpp used to recompute video_frame_gsd from the cached
// altitude) that happened to agree but had no single source of truth. Consolidated here
// so a real per-video camera model (e.g. from telemetry/EXIF) has one place to override.
namespace CameraModel {

// Assumed horizontal field of view until real per-video camera metadata is available.
constexpr double kDefaultHfovDeg = 70.0;

// Real-world width (metres) of a frame's footprint at the given altitude.
inline double frameFootprintMeters(double altitude_m, double hfov_deg = kDefaultHfovDeg) {
    double hfov_rad = hfov_deg * M_PI / 180.0;
    return 2.0 * altitude_m * std::tan(hfov_rad / 2.0);
}

// Inverse of frameFootprintMeters: altitude (metres) implied by a known footprint width.
inline double altitudeFromFootprintMeters(double footprint_width_m, double hfov_deg = kDefaultHfovDeg) {
    double hfov_rad = hfov_deg * M_PI / 180.0;
    return (footprint_width_m / 2.0) / std::tan(hfov_rad / 2.0);
}

// Ground-sample-distance (metres/pixel) of a frame of the given pixel width.
inline double groundSampleDistance(double altitude_m, int frame_width_px, double hfov_deg = kDefaultHfovDeg) {
    return frameFootprintMeters(altitude_m, hfov_deg) / frame_width_px;
}

} // namespace CameraModel
