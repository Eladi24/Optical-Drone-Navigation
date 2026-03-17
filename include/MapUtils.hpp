#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <curl/curl.h>
#include <fstream>
/**
 * @file MapUtils.hpp
 * @brief Utilities for fetching, caching, and working with Google Maps API images
 */

/**
 * CURL callback function to collect binary data into a byte buffer
 * Used internally by fetchMapImage()
 */
size_t WriteToVector(void *contents, size_t size, size_t nmemb, void *userp);

/**
 * Fetch a map image from a URL using libcurl
 * 
 * @param url Full URL to fetch (typically Google Maps Static API URL)
 * @return cv::Mat containing the downloaded image, or empty Mat on failure
 */
cv::Mat fetchMapImage(const std::string& url);

/**
 * Load map from disk cache if it exists, otherwise fetch from API and save
 * This is the recommended way to get map images to avoid unnecessary API calls
 * 
 * @param filename Path to the cached image file (e.g., "Images/map_clean_haifa.png")
 * @param url Full URL to fetch if cache miss
 * @param verbose If true, print status messages about cache hits/misses
 * @return cv::Mat containing the map image, or empty Mat on failure
 */
cv::Mat loadOrFetchMapImage(const std::string& filename, const std::string& url, bool verbose = true);

/**
 * Calculate meters per pixel for Web Mercator projection at given latitude and zoom
 * 
 * @param lat_deg Latitude in degrees
 * @param zoom Google Maps zoom level (0-21)
 * @return Meters per pixel at the specified latitude and zoom
 */
double metersPerPixel(double lat_deg, int zoom);

/**
 * Choose an appropriate Google Maps zoom level to achieve a target span in meters
 * 
 * @param lat_deg Latitude in degrees (affects meters per pixel calculation)
 * @param target_meters Desired map span in meters
 * @param pixel_span Number of pixels across the map
 * @return Optimal zoom level (clamped to 0-21)
 */
int chooseZoomForSpan(double lat_deg, double target_meters, int pixel_span);

/**
 * Read Google Maps API key from config file
 * 
 * @param configPath Path to config.ini file (default: "config.ini")
 * @return API key string, or empty string on failure
 */
std::string readApiKey(const std::string& configPath = "config.ini");