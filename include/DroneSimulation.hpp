#pragma once
#include <opencv2/opencv.hpp>
#include <random>

class DroneSimulation
{
    private:
        double lat, lng;               // Current position
        double heading;                 // Current heading in degrees (0 = North, 90 = East)
        double speed;                   // Speed in meters per second
        double drift;                   // Random drift in heading (degrees)
        double meters_per_degree_lat;  // Conversion factors
        double meters_per_degree_lng;

        cv::Mat map;                    // Full map for simulation
        cv::Mat drone_view;             // Current view from drone
        int view_size;                  // Size of drone's view in pixels
        // Flight history
        std::vector<std::pair<double, double>> flight_path;
        

    public:
        DroneSimulation(double start_lat, double start_lng, double target_heading,
                        double speed_mps, double heading_drift,
                        double meters_per_deg_lat, double meters_per_deg_lng,
                        const cv::Mat &map_image, int view_size_px)
            : lat(start_lat), lng(start_lng),
              heading(target_heading), speed(speed_mps), drift(heading_drift),
              meters_per_degree_lat(meters_per_deg_lat),
              meters_per_degree_lng(meters_per_deg_lng),
              map(map_image.clone()), view_size(view_size_px)
        {

            // Initialize drone's view as black image
            drone_view = cv::Mat(view_size, view_size, CV_8UC3, cv::Scalar(0, 0, 0));

            // Record starting position
            flight_path.push_back(std::make_pair(lat, lng));
        }
        void step(double dt_seconds, double correction_degrees = 0.0);
        void updateView(double center_lat, double center_lng, int center_x, int center_y, double mpp);
        cv::Point latLngToPixel(double lat, double lng, double center_lat, double center_lng, 
                          int center_x, int center_y, double mpp);
        std::pair<double, double> getPosition() const { return std::make_pair(lat, lng); }
        cv::Mat getCurrentView() const { return drone_view; }
        double getHeading() const { return heading; }
        const std::vector<std::pair<double, double>>& getFlightPath() const { return flight_path; }

        void setHeading(double new_heading) { heading = new_heading; }
        
};

struct CoordinateHash {
    std::size_t operator()(const std::pair<double, double>& coord) const {
        return std::hash<double>()(coord.first) ^ (std::hash<double>()(coord.second) << 1);
    }
};