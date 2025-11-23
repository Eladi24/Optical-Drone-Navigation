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
    
    // 🆕 Extract location and path type from config.name
    // config.name format: "diagonal_manhattan" or "zigzag_jerusalem"
    size_t underscore_pos = config.name.find('_');
    std::string path_type = config.name.substr(0, underscore_pos);  // "diagonal" or "zigzag"
    std::string location = config.name.substr(underscore_pos + 1);  // "manhattan" or "jerusalem"
    
    std::cout << "\nStarting drone flight simulation for " << config.name << "..." << std::endl;
    runDroneSimulation(clean_map, crops, config.waypoints,
                      meters_per_degree_lat, meters_per_degree_lng,
                      lat, lng, center_x, center_y, mpp, algorithm,
                      location, path_type);  // 🆕 PASS NEW PARAMETERS
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
    
    // ==================== JERUSALEM SIMULATION ====================
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "🚁 JERUSALEM SIMULATION" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    const double lat_jer = 31.7767;
    const double lng_jer = 35.2345;
    const int width = 640;
    const int height = 640;
    const int scale = 2;
    const std::string maptype = "satellite";

    const double target_span_m = 1000.0;
    const int effective_px = width * scale;
    int zoom_jer = chooseZoomForSpan(lat_jer, target_span_m, effective_px);

    const double lat_radians_jer = lat_jer * M_PI / 180.0;
    const double meters_per_degree_lat_jer = 111320.0;
    const double meters_per_degree_lng_jer = 111320.0 * std::cos(lat_radians_jer);

    const double lat_offset_start_jer = -250.0 / meters_per_degree_lat_jer;
    const double lng_offset_start_jer = -250.0 / meters_per_degree_lng_jer;

    const double path_length_m = 350.0;
    const double diagonal_component_m = path_length_m / sqrt(2.0);

    const double lat_offset_path_jer = diagonal_component_m / meters_per_degree_lat_jer;
    const double lng_offset_path_jer = diagonal_component_m / meters_per_degree_lng_jer;

    const double path_start_lat_jer = lat_jer + lat_offset_start_jer + lat_offset_path_jer;
    const double path_start_lng_jer = lng_jer + lng_offset_start_jer + lng_offset_path_jer;

    const double path_end_lat_jer = path_start_lat_jer + lat_offset_path_jer;
    const double path_end_lng_jer = path_start_lng_jer + lng_offset_path_jer;

    double mpp_jer = metersPerPixel(lat_jer, zoom_jer);
    double span_m_jer = mpp_jer * effective_px;
    std::cout << "Jerusalem - Chosen zoom=" << zoom_jer
              << " → mpp=" << mpp_jer
              << " → width span ≈ " << span_m_jer << " m\n";

    std::ostringstream clean_url_ss_jer;
    clean_url_ss_jer << "https://maps.googleapis.com/maps/api/staticmap?"
                     << "center=" << lat_jer << "," << lng_jer
                     << "&zoom=" << zoom_jer
                     << "&size=" << width << "x" << height
                     << "&maptype=" << maptype
                     << "&scale=" << scale
                     << "&key=" << apiKey;

    cv::Mat clean_map_jer = fetchMapImage(clean_url_ss_jer.str());
    if (clean_map_jer.empty())
    {
        std::cerr << "Failed to fetch Jerusalem clean map\n";
        return 1;
    }
    cv::imwrite("Images/map_clean_jerusalem.png", clean_map_jer);

    double pixels_per_100m_jer = 100.0 / mpp_jer;
    int crop_size_jer = static_cast<int>(std::round(pixels_per_100m_jer));
    int center_x_jer = (width * scale) / 2;
    int center_y_jer = (height * scale) / 2;

    // Diagonal Path - Jerusalem
    PathConfig diagonal_path_jer;
    diagonal_path_jer.name = "diagonal_jerusalem";
    diagonal_path_jer.color = "0x0000FF";
    diagonal_path_jer.start_marker_color = "red";
    diagonal_path_jer.end_marker_color = "green";
    diagonal_path_jer.waypoints = {{path_start_lat_jer, path_start_lng_jer}, {path_end_lat_jer, path_end_lng_jer}};
    
    for (int i = 0; i < 10; i++) {
        double t = i / 9.0;
        diagonal_path_jer.sample_points.push_back({
            path_start_lat_jer + t * (path_end_lat_jer - path_start_lat_jer),
            path_start_lng_jer + t * (path_end_lng_jer - path_start_lng_jer)
        });
    }
    
    processAndSimulatePath(diagonal_path_jer, clean_map_jer, apiKey, lat_jer, lng_jer, zoom_jer, 
                          width, height, scale, maptype, center_x_jer, center_y_jer, 
                          mpp_jer, crop_size_jer, meters_per_degree_lat_jer, 
                          meters_per_degree_lng_jer, algorithm);
    
    // Zigzag Path - Jerusalem
    PathConfig zigzag_path_jer;
    zigzag_path_jer.name = "zigzag_jerusalem";
    zigzag_path_jer.color = "0xFF0000";
    zigzag_path_jer.start_marker_color = "blue";
    zigzag_path_jer.end_marker_color = "purple";
    
    double step_size_jer = lat_offset_path_jer / 4.0;
    zigzag_path_jer.waypoints = {
        {path_start_lat_jer, path_start_lng_jer},
        {path_start_lat_jer + step_size_jer, path_start_lng_jer + step_size_jer},
        {path_start_lat_jer + 2 * step_size_jer, path_start_lng_jer},
        {path_start_lat_jer + 3 * step_size_jer, path_start_lng_jer + step_size_jer},
        {path_start_lat_jer + 4 * step_size_jer, path_start_lng_jer},
        {path_end_lat_jer, path_start_lng_jer + 0.5 * step_size_jer}
    };
    
    for (size_t i = 0; i < zigzag_path_jer.waypoints.size() - 1; i++) {
        const auto& start = zigzag_path_jer.waypoints[i];
        const auto& end = zigzag_path_jer.waypoints[i+1];
        int points_per_segment = (i < zigzag_path_jer.waypoints.size() - 2) ? 2 : 3;
        
        for (int j = 0; j < points_per_segment; j++) {
            double t = j / static_cast<double>(points_per_segment);
            zigzag_path_jer.sample_points.push_back({
                start.first + t * (end.first - start.first),
                start.second + t * (end.second - start.second)
            });
        }
    }
    zigzag_path_jer.sample_points.push_back(zigzag_path_jer.waypoints.back());
    
    processAndSimulatePath(zigzag_path_jer, clean_map_jer, apiKey, lat_jer, lng_jer, zoom_jer,
                          width, height, scale, maptype, center_x_jer, center_y_jer,
                          mpp_jer, crop_size_jer, meters_per_degree_lat_jer,
                          meters_per_degree_lng_jer, algorithm);

    // ==================== MANHATTAN SIMULATION ====================
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "🗽 MANHATTAN SIMULATION" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    const double lat_nyc = 40.7580;  // Central Park area
    const double lng_nyc = -73.9855;

    int zoom_nyc = chooseZoomForSpan(lat_nyc, target_span_m, effective_px);

    const double lat_radians_nyc = lat_nyc * M_PI / 180.0;
    const double meters_per_degree_lat_nyc = 111320.0;
    const double meters_per_degree_lng_nyc = 111320.0 * std::cos(lat_radians_nyc);

    const double lat_offset_start_nyc = -250.0 / meters_per_degree_lat_nyc;
    const double lng_offset_start_nyc = -250.0 / meters_per_degree_lng_nyc;

    const double lat_offset_path_nyc = diagonal_component_m / meters_per_degree_lat_nyc;
    const double lng_offset_path_nyc = diagonal_component_m / meters_per_degree_lng_nyc;

    const double path_start_lat_nyc = lat_nyc + lat_offset_start_nyc + lat_offset_path_nyc;
    const double path_start_lng_nyc = lng_nyc + lng_offset_start_nyc + lng_offset_path_nyc;

    const double path_end_lat_nyc = path_start_lat_nyc + lat_offset_path_nyc;
    const double path_end_lng_nyc = path_start_lng_nyc + lng_offset_path_nyc;

    double mpp_nyc = metersPerPixel(lat_nyc, zoom_nyc);
    double span_m_nyc = mpp_nyc * effective_px;
    std::cout << "Manhattan - Chosen zoom=" << zoom_nyc
              << " → mpp=" << mpp_nyc
              << " → width span ≈ " << span_m_nyc << " m\n";

    std::ostringstream clean_url_ss_nyc;
    clean_url_ss_nyc << "https://maps.googleapis.com/maps/api/staticmap?"
                     << "center=" << lat_nyc << "," << lng_nyc
                     << "&zoom=" << zoom_nyc
                     << "&size=" << width << "x" << height
                     << "&maptype=" << maptype
                     << "&scale=" << scale
                     << "&key=" << apiKey;

    cv::Mat clean_map_nyc = fetchMapImage(clean_url_ss_nyc.str());
    if (clean_map_nyc.empty())
    {
        std::cerr << "Failed to fetch Manhattan clean map\n";
        return 1;
    }
    cv::imwrite("Images/map_clean_manhattan.png", clean_map_nyc);

    double pixels_per_100m_nyc = 100.0 / mpp_nyc;
    int crop_size_nyc = static_cast<int>(std::round(pixels_per_100m_nyc));
    int center_x_nyc = (width * scale) / 2;
    int center_y_nyc = (height * scale) / 2;

    // Diagonal Path - Manhattan
    PathConfig diagonal_path_nyc;
    diagonal_path_nyc.name = "diagonal_manhattan";
    diagonal_path_nyc.color = "0x0000FF";
    diagonal_path_nyc.start_marker_color = "red";
    diagonal_path_nyc.end_marker_color = "green";
    diagonal_path_nyc.waypoints = {{path_start_lat_nyc, path_start_lng_nyc}, {path_end_lat_nyc, path_end_lng_nyc}};
    
    for (int i = 0; i < 10; i++) {
        double t = i / 9.0;
        diagonal_path_nyc.sample_points.push_back({
            path_start_lat_nyc + t * (path_end_lat_nyc - path_start_lat_nyc),
            path_start_lng_nyc + t * (path_end_lng_nyc - path_start_lng_nyc)
        });
    }
    
    processAndSimulatePath(diagonal_path_nyc, clean_map_nyc, apiKey, lat_nyc, lng_nyc, zoom_nyc, 
                          width, height, scale, maptype, center_x_nyc, center_y_nyc, 
                          mpp_nyc, crop_size_nyc, meters_per_degree_lat_nyc, 
                          meters_per_degree_lng_nyc, algorithm);
    
    // Zigzag Path - Manhattan
    PathConfig zigzag_path_nyc;
    zigzag_path_nyc.name = "zigzag_manhattan";
    zigzag_path_nyc.color = "0xFF0000";
    zigzag_path_nyc.start_marker_color = "blue";
    zigzag_path_nyc.end_marker_color = "purple";
    
    double step_size_nyc = lat_offset_path_nyc / 4.0;
    zigzag_path_nyc.waypoints = {
        {path_start_lat_nyc, path_start_lng_nyc},
        {path_start_lat_nyc + step_size_nyc, path_start_lng_nyc + step_size_nyc},
        {path_start_lat_nyc + 2 * step_size_nyc, path_start_lng_nyc},
        {path_start_lat_nyc + 3 * step_size_nyc, path_start_lng_nyc + step_size_nyc},
        {path_start_lat_nyc + 4 * step_size_nyc, path_start_lng_nyc},
        {path_end_lat_nyc, path_start_lng_nyc + 0.5 * step_size_nyc}
    };
    
    for (size_t i = 0; i < zigzag_path_nyc.waypoints.size() - 1; i++) {
        const auto& start = zigzag_path_nyc.waypoints[i];
        const auto& end = zigzag_path_nyc.waypoints[i+1];
        int points_per_segment = (i < zigzag_path_nyc.waypoints.size() - 2) ? 2 : 3;
        
        for (int j = 0; j < points_per_segment; j++) {
            double t = j / static_cast<double>(points_per_segment);
            zigzag_path_nyc.sample_points.push_back({
                start.first + t * (end.first - start.first),
                start.second + t * (end.second - start.second)
            });
        }
    }
    zigzag_path_nyc.sample_points.push_back(zigzag_path_nyc.waypoints.back());
    
    processAndSimulatePath(zigzag_path_nyc, clean_map_nyc, apiKey, lat_nyc, lng_nyc, zoom_nyc,
                          width, height, scale, maptype, center_x_nyc, center_y_nyc,
                          mpp_nyc, crop_size_nyc, meters_per_degree_lat_nyc,
                          meters_per_degree_lng_nyc, algorithm);

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "✅ ALL SIMULATIONS COMPLETE!" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return 0;
}
