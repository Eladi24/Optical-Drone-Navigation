#include "DatasetVideoAssembler.hpp"

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

std::string zeroPadded(int n, int width) {
    std::ostringstream ss;
    ss << std::setw(width) << std::setfill('0') << n;
    return ss.str();
}

} // namespace

bool assembleDatasetVideo(const std::string& image_dir,
                          const std::string& prefix,
                          int frame_count,
                          const std::string& output_path) {
    std::ifstream existing(output_path);
    if (existing.good()) {
        std::cout << "assembleDatasetVideo: using cached " << output_path << std::endl;
        return true;
    }

    // Nominal frame rate only -- see header comment. Real timing is carried
    // separately via TelemetryImporter's telemetry CSV.
    constexpr double kNominalFps = 5.0;

    cv::VideoWriter writer;
    bool writer_opened = false;
    int  written = 0;

    for (int i = 1; i <= frame_count; ++i) {
        std::string path = image_dir + "/" + prefix + "_" + zeroPadded(i, 4) + ".JPG";
        cv::Mat frame = cv::imread(path);
        if (frame.empty()) {
            std::cerr << "assembleDatasetVideo: could not read " << path << ", skipping." << std::endl;
            continue;
        }

        if (!writer_opened) {
            writer.open(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                        kNominalFps, frame.size(), true);
            if (!writer.isOpened()) {
                std::cerr << "assembleDatasetVideo: could not open writer for " << output_path << std::endl;
                return false;
            }
            writer_opened = true;
        }

        writer.write(frame);
        ++written;
    }

    if (!writer_opened) {
        std::cerr << "assembleDatasetVideo: no frames were readable from " << image_dir << std::endl;
        return false;
    }

    std::cout << "assembleDatasetVideo: wrote " << written << "/" << frame_count
              << " frames -> " << output_path << std::endl;
    return true;
}
