#include <curl/curl.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <iomanip>
#include "FlightSimulation.hpp"
#include "CoordinateUtils.hpp"
#include <fstream>

// Write callback to collect binary data into a byte buffer
size_t WriteToVector(void *contents, size_t size, size_t nmemb, void *userp)
{
    size_t total = size * nmemb;
    auto *buf = static_cast<std::vector<unsigned char> *>(userp);
    unsigned char *c = static_cast<unsigned char *>(contents);
    buf->insert(buf->end(), c, c + total);
    return total;
}

// Web Mercator meters-per-pixel at latitude phi (radians) and zoom z
static double metersPerPixel(double lat_deg, int z)
{
    const double R = 6378137.0;
    const double phi = lat_deg * M_PI / 180.0;
    return std::cos(phi) * 2.0 * M_PI * R / (256.0 * std::pow(2.0, z));
}

// Choose an integer zoom so that the image spans approximately target_meters across
static int chooseZoomForSpan(double lat_deg, double target_meters, int pixel_span)
{
    const double R = 6378137.0;
    const double phi = lat_deg * M_PI / 180.0;
    const double target_mpp = target_meters / pixel_span;
    double z_real = std::log2(std::cos(phi) * 2.0 * M_PI * R / (256.0 * target_mpp));
    int z = (int)std::round(z_real);
    if (z < 0)
        z = 0;
    if (z > 21)
        z = 21;
    return z;
}

// Function to fetch map image using libcurl
cv::Mat fetchMapImage(const std::string &url)
{
    CURL *curl = curl_easy_init();
    if (!curl)
    {
        std::cerr << "Failed to init CURL\n";
        return cv::Mat();
    }

    std::vector<unsigned char> buffer;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteToVector);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK)
    {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << "\n";
        return cv::Mat();
    }

    if (buffer.empty())
    {
        std::cerr << "Got empty image response\n";
        return cv::Mat();
    }

    cv::Mat data_mat(1, static_cast<int>(buffer.size()), CV_8UC1, buffer.data());
    cv::Mat img = cv::imdecode(data_mat, cv::IMREAD_COLOR);

    if (img.empty())
    {
        std::cerr << "Failed to decode image\n";
    }

    return img;
}

std::string readApiKey(const std::string& configPath = "config.ini") {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open config file at " << configPath << std::endl;
        std::cerr << "Please copy config.ini.sample to config.ini and add your API key" << std::endl;
        return "";
    }
    
    std::string line, key;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }
        
        if (line[0] == '[') {
            continue;
        }
        
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string name = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            name.erase(0, name.find_first_not_of(" \t"));
            name.erase(name.find_last_not_of(" \t") + 1);
            
            if (name == "key") {
                return value;
            }
        }
    }
    
    std::cerr << "Error: API key not found in config file" << std::endl;
    return "";
}

struct PathConfig {
    std::string name;
    std::string color;
    std::string start_marker_color;
    std::string end_marker_color;
    std::vector<std::pair<double, double>> waypoints;
    std::vector<std::pair<double, double>> sample_points;
};

void processAndSimulatePath(
    const PathConfig& config,
    const cv::Mat& clean_map,
    const std::string& apiKey,
    double lat, double lng, int zoom, int width, int height, 
    int scale, const std::string& maptype,
    int center_x, int center_y, double mpp, int crop_size,
    double meters_per_degree_lat, double meters_per_degree_lng,
    PositionAlgorithm algorithm)
{
    // Generate map URL
    std::ostringstream url_ss;
    url_ss << "https://maps.googleapis.com/maps/api/staticmap?"
           << "center=" << lat << "," << lng
           << "&zoom=" << zoom << "&size=" << width << "x" << height
           << "&maptype=" << maptype << "&scale=" << scale
           << "&path=color:" << config.color << "|weight:5";
    
    for (const auto& point : config.waypoints) {
        url_ss << "|" << point.first << "," << point.second;
    }
    
    url_ss << "&markers=color:" << config.start_marker_color << "|label:S|" 
           << config.waypoints.front().first << "," << config.waypoints.front().second
           << "&markers=color:" << config.end_marker_color << "|label:E|"
           << config.waypoints.back().first << "," << config.waypoints.back().second;
    
    url_ss << "&markers=color:yellow|size:small";
    for (const auto& point : config.sample_points) {
        url_ss << "|" << point.first << "," << point.second;
    }
    url_ss << "&key=" << apiKey;
    
    // Fetch and save map
    cv::Mat path_map = fetchMapImage(url_ss.str());
    if (path_map.empty()) {
        std::cerr << "Failed to fetch " << config.name << " map\n";
        return;
    }
    cv::imwrite("Images/map_" + config.name + ".png", path_map);
    
    // Generate crops
    std::vector<ReferenceCrop> crops;
    for (const auto& point : config.sample_points) {
        cv::Point pt = CoordinateUtils::latLngToPixel(
            point.first, point.second,
            lat, lng, center_x, center_y, mpp,
            meters_per_degree_lat, meters_per_degree_lng);
        
        cv::Rect crop_rect(pt.x - crop_size / 2, pt.y - crop_size / 2, 
                          crop_size, crop_size);
        crop_rect = crop_rect & cv::Rect(0, 0, clean_map.cols, clean_map.rows);
        
        if (crop_rect.width > 0 && crop_rect.height > 0) {
            cv::Mat cropped = clean_map(crop_rect).clone();
            crops.push_back({point, cropped});
        }
    }
    
    // Display
    cv::namedWindow("Map with " + config.name + " Path", cv::WINDOW_NORMAL);
    cv::imshow("Map with " + config.name + " Path", path_map);
    
    for (int i = 0; i < crops.size(); i++) {
        std::ostringstream window_name;
        window_name << config.name << " Crop " << i << " - Coordinates: "
                    << std::fixed << std::setprecision(6)
                    << crops[i].coordinates.first << ", " << crops[i].coordinates.second;
        cv::namedWindow(window_name.str(), cv::WINDOW_AUTOSIZE);
        cv::imshow(window_name.str(), crops[i].image);
    }
    
    std::cout << "Displaying " << crops.size() << " crops from " << config.name << " path" << std::endl;
    std::cout << "Press any key to continue" << std::endl;
    cv::waitKey(0);
    
    // Match and simulate
    std::cout << "\nPerforming template matching for " << config.name << " path crops..." << std::endl;
    matchCropsOnMap(clean_map, crops);
    
    std::cout << "\nStarting drone flight simulation for " << config.name << "..." << std::endl;
    runDroneSimulation(clean_map, crops, config.waypoints,
                      meters_per_degree_lat, meters_per_degree_lng,
                      lat, lng, center_x, center_y, mpp, algorithm);
}

int main(int argc, char** argv)
{
    PositionAlgorithm algorithm = PositionAlgorithm::HYBRID;
    
    if (argc > 1) {
        std::string algo_str = argv[1];
        if (algo_str == "template" || algo_str == "TEMPLATE")
            algorithm = PositionAlgorithm::TEMPLATE;
        else if (algo_str == "orb" || algo_str == "ORB")
            algorithm = PositionAlgorithm::ORB;
        else if (algo_str == "sift" || algo_str == "SIFT")
            algorithm = PositionAlgorithm::SIFT;
        else if (algo_str == "surf" || algo_str == "SURF")
            algorithm = PositionAlgorithm::SURF;
        else if (algo_str == "smoothed" || algo_str == "SMOOTHED")
            algorithm = PositionAlgorithm::SMOOTHED;
        else if (algo_str == "hybrid" || algo_str == "HYBRID")
            algorithm = PositionAlgorithm::HYBRID;
    }
    
    const std::string apiKey = readApiKey();
    if (apiKey.empty()) {
        std::cerr << "Failed to read API key. Please check your config.ini file." << std::endl;
        return 1;
    }
    
    const double lat = 31.7767;
    const double lng = 35.2345;
    const int width = 640;
    const int height = 640;
    const int scale = 2;
    const std::string maptype = "satellite";

    const double target_span_m = 1000.0;
    const int effective_px = width * scale;
    int zoom = chooseZoomForSpan(lat, target_span_m, effective_px);

    const double lat_radians = lat * M_PI / 180.0;
    const double meters_per_degree_lat = 111320.0;
    const double meters_per_degree_lng = 111320.0 * std::cos(lat_radians);

    const double lat_offset_start = -250.0 / meters_per_degree_lat;
    const double lng_offset_start = -250.0 / meters_per_degree_lng;

    const double path_length_m = 350.0;
    const double diagonal_component_m = path_length_m / sqrt(2.0);

    const double lat_offset_path = diagonal_component_m / meters_per_degree_lat;
    const double lng_offset_path = diagonal_component_m / meters_per_degree_lng;

    const double path_start_lat = lat + lat_offset_start + lat_offset_path;
    const double path_start_lng = lng + lng_offset_start + lng_offset_path;

    const double path_end_lat = path_start_lat + lat_offset_path;
    const double path_end_lng = path_start_lng + lng_offset_path;

    double mpp = metersPerPixel(lat, zoom);
    double span_m = mpp * effective_px;
    std::cerr << "Chosen zoom=" << zoom
              << " → mpp=" << mpp
              << " → width span ≈ " << span_m << " m\n";

    std::ostringstream clean_url_ss;
    clean_url_ss << "https://maps.googleapis.com/maps/api/staticmap?"
                 << "center=" << lat << "," << lng
                 << "&zoom=" << zoom
                 << "&size=" << width << "x" << height
                 << "&maptype=" << maptype
                 << "&scale=" << scale
                 << "&key=" << apiKey;

    cv::Mat clean_map = fetchMapImage(clean_url_ss.str());
    if (clean_map.empty())
    {
        std::cerr << "Failed to fetch clean map\n";
        return 1;
    }
    cv::imwrite("Images/map_clean.png", clean_map);

    char url[2048];
    std::snprintf(url, sizeof(url),
                  "https://maps.googleapis.com/maps/api/staticmap?"
                  "center=%f,%f&zoom=%d&size=%dx%d&maptype=%s&scale=%d"
                  "&path=color:0x0000FF|weight:5|%f,%f|%f,%f"
                  "&markers=color:red|label:S|%f,%f"
                  "&markers=color:green|label:E|%f,%f"
                  "&key=%s",
                  lat, lng, zoom, width, height, maptype.c_str(), scale,
                  path_start_lat, path_start_lng, path_end_lat, path_end_lng,
                  path_start_lat, path_start_lng,
                  path_end_lat, path_end_lng,
                  apiKey.c_str());

    std::vector<std::pair<double, double>> sample_points;
    for (int i = 0; i < 10; i++)
    {
        double t = i / 9.0;
        double point_lat = path_start_lat + t * (path_end_lat - path_start_lat);
        double point_lng = path_start_lng + t * (path_end_lng - path_start_lng);
        sample_points.push_back(std::make_pair(point_lat, point_lng));
    }

    std::ostringstream marked_url_ss;
    marked_url_ss << "https://maps.googleapis.com/maps/api/staticmap?"
                  << "center=" << lat << "," << lng
                  << "&zoom=" << zoom
                  << "&size=" << width << "x" << height
                  << "&maptype=" << maptype
                  << "&scale=" << scale
                  << "&path=color:0x0000FF|weight:5|" << path_start_lat << "," << path_start_lng
                  << "|" << path_end_lat << "," << path_end_lng
                  << "&markers=color:red|label:S|" << path_start_lat << "," << path_start_lng
                  << "&markers=color:green|label:E|" << path_end_lat << "," << path_end_lng;

    marked_url_ss << "&markers=color:yellow|size:small";
    for (const auto &point : sample_points)
    {
        marked_url_ss << "|" << point.first << "," << point.second;
    }
    marked_url_ss << "&key=" << apiKey;

    cv::Mat marked_map = fetchMapImage(marked_url_ss.str());
    if (marked_map.empty())
    {
        std::cerr << "Failed to fetch marked map\n";
        return 1;
    }
    cv::imwrite("Images/map_marked.png", marked_map);

    double pixels_per_100m = 100.0 / mpp;
    int crop_size = static_cast<int>(std::round(pixels_per_100m));

    std::vector<ReferenceCrop> crops;

    int center_x = (width * scale) / 2;
    int center_y = (height * scale) / 2;

    for (int i = 0; i < sample_points.size(); i++)
    {
        const auto &point = sample_points[i];

        cv::Point pt = CoordinateUtils::latLngToPixel(
            point.first, point.second,
            lat, lng, center_x, center_y, mpp,
            meters_per_degree_lat, meters_per_degree_lng);

        cv::Rect crop_rect(
            pt.x - crop_size / 2,
            pt.y - crop_size / 2,
            crop_size,
            crop_size);

        crop_rect = crop_rect & cv::Rect(0, 0, clean_map.cols, clean_map.rows);

        if (crop_rect.width > 0 && crop_rect.height > 0)
        {
            cv::Mat cropped = clean_map(crop_rect).clone();
            crops.push_back({point, cropped});
        }
    }

    cv::namedWindow("Map with Path and Markers", cv::WINDOW_NORMAL);
    cv::imshow("Map with Path and Markers", marked_map);

    int i = 0;
    for (const auto &crop : crops)
    {
        std::ostringstream window_name;
        window_name << "Crop " << i << " - Coordinates: "
                    << std::fixed << std::setprecision(6)
                    << crop.coordinates.first << ", " << crop.coordinates.second;

        cv::namedWindow(window_name.str(), cv::WINDOW_AUTOSIZE);
        cv::imshow(window_name.str(), crop.image);
        i++;
    }

    std::cout << "Displaying " << crops.size() << " crops from the path" << std::endl;
    std::cout << "Press any key to close all windows" << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    matchCropsOnMap(clean_map, crops);

    std::cout << "\nStarting drone flight simulation..." << std::endl;
    std::vector<std::pair<double, double>> waypoints = {
        {path_start_lat, path_start_lng},
        {path_end_lat, path_end_lng}
    };
    runDroneSimulation(
        clean_map, crops,
        waypoints,
        meters_per_degree_lat, meters_per_degree_lng,
        lat, lng,
        (width * scale) / 2, (height * scale) / 2,
        mpp,
        algorithm
    );
    
    // Diagonal Path
    PathConfig diagonal_path;
    diagonal_path.name = "diagonal";
    diagonal_path.color = "0x0000FF";
    diagonal_path.start_marker_color = "red";
    diagonal_path.end_marker_color = "green";
    diagonal_path.waypoints = {{path_start_lat, path_start_lng}, {path_end_lat, path_end_lng}};
    
    for (int i = 0; i < 10; i++) {
        double t = i / 9.0;
        diagonal_path.sample_points.push_back({
            path_start_lat + t * (path_end_lat - path_start_lat),
            path_start_lng + t * (path_end_lng - path_start_lng)
        });
    }
    
    processAndSimulatePath(diagonal_path, clean_map, apiKey, lat, lng, zoom, 
                          width, height, scale, maptype, center_x, center_y, 
                          mpp, crop_size, meters_per_degree_lat, 
                          meters_per_degree_lng, algorithm);
    
    // Zigzag Path
    PathConfig zigzag_path;
    zigzag_path.name = "zigzag";
    zigzag_path.color = "0xFF0000";
    zigzag_path.start_marker_color = "blue";
    zigzag_path.end_marker_color = "purple";
    
    // Build zigzag waypoints
    double step_size = lat_offset_path / 4.0;
    zigzag_path.waypoints = {
        {path_start_lat, path_start_lng},
        {path_start_lat + step_size, path_start_lng + step_size},
        {path_start_lat + 2 * step_size, path_start_lng},
        {path_start_lat + 3 * step_size, path_start_lng + step_size},
        {path_start_lat + 4 * step_size, path_start_lng},
        {path_end_lat, path_start_lng + 0.5 * step_size}
    };
    
    // Generate zigzag samples
    for (size_t i = 0; i < zigzag_path.waypoints.size() - 1; i++) {
        const auto& start = zigzag_path.waypoints[i];
        const auto& end = zigzag_path.waypoints[i+1];
        int points_per_segment = (i < zigzag_path.waypoints.size() - 2) ? 2 : 3;
        
        for (int j = 0; j < points_per_segment; j++) {
            double t = j / static_cast<double>(points_per_segment);
            zigzag_path.sample_points.push_back({
                start.first + t * (end.first - start.first),
                start.second + t * (end.second - start.second)
            });
        }
    }
    zigzag_path.sample_points.push_back(zigzag_path.waypoints.back());
    
    processAndSimulatePath(zigzag_path, clean_map, apiKey, lat, lng, zoom,
                          width, height, scale, maptype, center_x, center_y,
                          mpp, crop_size, meters_per_degree_lat,
                          meters_per_degree_lng, algorithm);
    
    return 0;
}
