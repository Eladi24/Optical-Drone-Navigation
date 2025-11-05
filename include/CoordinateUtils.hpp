// CoordinateUtils.hpp
#pragma once
#include <utility>
#include <opencv2/opencv.hpp>

struct CoordinateHash {
    std::size_t operator()(const std::pair<double, double> &p) const {
        auto h1 = std::hash<double>{}(p.first);
        auto h2 = std::hash<double>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class CoordinateUtils {
public:
    static cv::Point latLngToPixel(double lat, double lng,
                                   double center_lat, double center_lng,
                                   int center_x, int center_y,
                                   double mpp, double meters_per_degree_lat,
                                   double meters_per_degree_lng);
    
    static double calculateDistance(const std::pair<double, double> &pos1,
                                   const std::pair<double, double> &pos2,
                                   double meters_per_degree_lat,
                                   double meters_per_degree_lng);
    
    static double calculateHeading(const std::pair<double, double> &from,
                                  const std::pair<double, double> &to,
                                  double meters_per_degree_lat,
                                  double meters_per_degree_lng);
};