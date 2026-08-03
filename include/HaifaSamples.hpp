#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Per-sample configuration for the Haifa video pipeline.
//
// Single source of truth: both Main.cpp (navigation) and the GroundTruth
// annotation tool must use the identical center_lat/center_lng for a sample,
// otherwise pixel<->lat/lng conversions between the two tools would silently
// disagree and ground-truth annotations would not line up with telemetry.
// ---------------------------------------------------------------------------
struct HaifaSampleConfig {
    std::string video_path;
    std::string map_path;
    std::string location_name;
    std::string sample_name;
    double center_lat;
    double center_lng;
    std::string map_url_key;  // used to re-fetch if map is missing
};

// NOTE on Sample 3 coordinates:
// The on-screen caption in the video itself reads "الكريوت" (Al-Krayot) and
// its subtitle names Kiryat Yam, Kiryat Motzkin, Kiryat Ata, Kiryat Bialik,
// Kiryat Haim -- the Krayot cluster of towns northeast of Haifa, across the
// bay. The previous guess (32.8300, 34.9900) sits in Haifa proper, ~7-8km
// away, which is far larger than the 1500m map window the pipeline fetches --
// GlobalLocator would have been searching entirely the wrong area.
// The coordinates below are a best-effort correction (centered roughly on
// the Krayot cluster) and should still be verified precisely against
// Google Maps using a landmark visible in the video.
inline std::vector<HaifaSampleConfig> getHaifaSamples() {
    return {
        { "Haifa Samples/Sample 1.mp4", "Images/map_clean_haifa_north.png",
          "haifa_north", "sample1", 32.8350, 34.9790, "" },
        { "Haifa Samples/Sample 2.mp4", "Images/map_clean_haifa_east.png",
          "haifa_east",  "sample2", 32.8280, 34.9980, "" },
        { "Haifa Samples/Sample 3.mp4", "Images/map_clean_haifa_krayot.png",
          "haifa_krayot", "sample3", 32.8330, 35.0750, "" },
    };
}
