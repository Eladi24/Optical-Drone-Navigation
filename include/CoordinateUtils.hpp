// CoordinateUtils.hpp
#ifndef COORDINATEUTILS_HPP
#define COORDINATEUTILS_HPP
#include <utility>
#include <opencv2/opencv.hpp>


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
    
    static std::pair<double, double> pixelToLatLng(int pixel_x, int pixel_y,
                                                   double center_lat, double center_lng,
                                                   int center_x, int center_y,
                                                   double mpp, double meters_per_degree_lat,
                                                   double meters_per_degree_lng);
};
#endif // COORDINATEUTILS_HPP