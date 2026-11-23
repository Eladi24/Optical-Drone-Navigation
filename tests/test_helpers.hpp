#pragma once
// Shared helpers for the GoogleTest suite: procedural imagery (no real imagery in the repo,
// so no licensing questions) and a scoped temp working directory for code that writes to
// paths relative to the cwd (e.g. TelemetryImporter -> "CSV Files/").
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>

namespace testutil {

constexpr double kMetersPerDegLat = 111320.0;
inline double metersPerDegLng(double lat_deg) { return kMetersPerDegLat * std::cos(lat_deg * M_PI / 180.0); }

// Deterministic, feature-rich BGR texture: noisy background plus many random filled
// rectangles, circles and lines. Corners everywhere, so ORB/XFeat/DISK all find plenty.
inline cv::Mat makeTexture(int width, int height, uint64_t seed = 12345)
{
    cv::RNG rng(seed);
    cv::Mat img(height, width, CV_8UC3);
    rng.fill(img, cv::RNG::UNIFORM, cv::Scalar::all(60), cv::Scalar::all(110));
    const int n_shapes = width * height / 900;
    for (int i = 0; i < n_shapes; ++i) {
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        cv::Point p(rng.uniform(0, width), rng.uniform(0, height));
        switch (rng.uniform(0, 3)) {
            case 0: {
                cv::Point q = p + cv::Point(rng.uniform(4, 40), rng.uniform(4, 40));
                cv::rectangle(img, p, q, color, cv::FILLED);
                break;
            }
            case 1:
                cv::circle(img, p, rng.uniform(3, 18), color, cv::FILLED);
                break;
            default: {
                cv::Point q = p + cv::Point(rng.uniform(-50, 50), rng.uniform(-50, 50));
                cv::line(img, p, q, color, rng.uniform(1, 4));
                break;
            }
        }
    }
    return img;
}

// Creates a fresh temp directory, chdirs into it, and restores the old cwd and removes the
// directory on destruction.
class ScopedTempCwd {
public:
    ScopedTempCwd() : old_(std::filesystem::current_path())
    {
        std::string tmpl = (std::filesystem::temp_directory_path() / "dronenav_test_XXXXXX").string();
        char* made = mkdtemp(tmpl.data());
        if (!made) throw std::runtime_error("mkdtemp failed");
        dir_ = made;
        std::filesystem::current_path(dir_);
    }
    ~ScopedTempCwd()
    {
        std::error_code ec;
        std::filesystem::current_path(old_, ec);
        std::filesystem::remove_all(dir_, ec);
    }
    const std::filesystem::path& path() const { return dir_; }

private:
    std::filesystem::path old_;
    std::filesystem::path dir_;
};

}  // namespace testutil
