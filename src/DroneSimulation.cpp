#include "DroneSimulation.hpp"
#include "CoordinateUtils.hpp"

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
    // Convert drone position to pixel coordinates using CoordinateUtils
    cv::Point drone_pos = CoordinateUtils::latLngToPixel(
        lat, lng, 
        center_lat, center_lng, 
        center_x, center_y, 
        mpp, 
        meters_per_degree_lat, 
        meters_per_degree_lng);

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
    return CoordinateUtils::latLngToPixel(
        lat, lng,
        center_lat, center_lng,
        center_x, center_y,
        mpp,
        meters_per_degree_lat,
        meters_per_degree_lng);
}

