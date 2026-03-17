#include "MapUtils.hpp"

// Write callback to collect binary data into a byte buffer
size_t WriteToVector(void *contents, size_t size, size_t nmemb, void *userp)
{
    size_t total = size * nmemb;
    auto *buf = static_cast<std::vector<unsigned char> *>(userp);
    unsigned char *c = static_cast<unsigned char *>(contents);
    buf->insert(buf->end(), c, c + total);
    return total;
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
    long response_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
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
        std::cerr << "\n=== IMAGE FETCH ERROR ===\n";
        std::cerr << "HTTP Status Code: " << response_code << "\n";
        std::cerr << "Failed to decode image. The server likely returned an error message instead of an image.\n";
        
        // NEW: Convert the raw buffer to a string and print it so we can read the Google API error
        std::string error_msg(buffer.begin(), buffer.end());
        std::cerr << "Raw Server Response:\n" << error_msg << "\n";
        std::cerr << "=========================\n\n";
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

// Web Mercator meters-per-pixel at latitude phi (radians) and zoom z
static double metersPerPixel(double lat_deg, int z)
{
    const double R = 6378137.0;
    const double phi = lat_deg * M_PI / 180.0;
    return std::cos(phi) * 2.0 * M_PI * R / (256.0 * std::pow(2.0, z));
}

cv::Mat loadOrFetchMapImage(const std::string& filename, const std::string& url, bool verbose = true)
{
    // First, try to load from disk
    cv::Mat img = cv::imread(filename);
    
    if (!img.empty()) {
        if (verbose) {
            std::cout << "✓ Loaded cached map: " << filename << std::endl;
        }
        return img;
    }
    
    // File doesn't exist, fetch from API
    if (verbose) {
        std::cout << "⬇️  Fetching map from Google Maps API..." << std::endl;
    }
    
    img = fetchMapImage(url);
    
    if (!img.empty()) {
        cv::imwrite(filename, img);
        if (verbose) {
            std::cout << "✓ Saved map to: " << filename << std::endl;
        }
    } else {
        if (verbose) {
            std::cerr << "❌ Failed to fetch map from API" << std::endl;
        }
    }
    
    return img;
}