#include "CoordinateUtils.hpp"
#include <cmath>

cv::Point CoordinateUtils::latLngToPixel(double lat, double lng,
                                         double center_lat, double center_lng,
                                         int center_x, int center_y,
                                         double mpp, double meters_per_degree_lat,
                                         double meters_per_degree_lng)
{
    double delta_lat = lat - center_lat;
    double delta_lng = lng - center_lng;
    
    double delta_meters_y = delta_lat * meters_per_degree_lat;
    double delta_meters_x = delta_lng * meters_per_degree_lng;
    
    int pixel_x = center_x + static_cast<int>(delta_meters_x / mpp);
    int pixel_y = center_y - static_cast<int>(delta_meters_y / mpp);
    
    return cv::Point(pixel_x, pixel_y);
}

double CoordinateUtils::calculateDistance(const std::pair<double, double> &pos1,
                                         const std::pair<double, double> &pos2,
                                         double meters_per_degree_lat,
                                         double meters_per_degree_lng)
{
    double delta_lat = (pos2.first - pos1.first) * meters_per_degree_lat;
    double delta_lng = (pos2.second - pos1.second) * meters_per_degree_lng;
    
    return std::sqrt(delta_lat * delta_lat + delta_lng * delta_lng);
}

double CoordinateUtils::calculateHeading(const std::pair<double, double> &from,
                                        const std::pair<double, double> &to,
                                        double meters_per_degree_lat,
                                        double meters_per_degree_lng)
{
    double delta_lng = to.second - from.second;
    double delta_lat = to.first - from.first;
    
    double heading = std::atan2(delta_lng * meters_per_degree_lng,
                                delta_lat * meters_per_degree_lat) * 180.0 / M_PI;
    
    // Normalize to [0, 360)
    if (heading < 0)
        heading += 360.0;
    
    return heading;
}

std::pair<double, double> CoordinateUtils::pixelToLatLng(int pixel_x, int pixel_y,
                                                         double center_lat, double center_lng,
                                                         int center_x, int center_y,
                                                         double mpp, double meters_per_degree_lat,
                                                         double meters_per_degree_lng)
{
    // Convert pixel offsets back to meters
    double delta_meters_x = (pixel_x - center_x) * mpp;
    // Note: Y is inverted because pixel coordinates go down, but latitude goes up!
    double delta_meters_y = (center_y - pixel_y) * mpp; 
    
    // Convert meters back to degrees
    double delta_lat = delta_meters_y / meters_per_degree_lat;
    double delta_lng = delta_meters_x / meters_per_degree_lng;
    
    return {center_lat + delta_lat, center_lng + delta_lng};
}