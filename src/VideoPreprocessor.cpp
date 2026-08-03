#include "VideoPreprocessor.hpp"
#include <iostream>
#include <fstream>
#include <numeric>
#include <cmath>
#include <functional>
#include <sys/stat.h>
#include <filesystem>

// ---------------------------------------------------------------------------
// Interactive mask editor state
// ---------------------------------------------------------------------------
struct MaskEditorState {
    cv::Mat first_frame;
    cv::Mat current_mask;   // working copy (255=terrain, 0=excluded)
    cv::Mat auto_mask;      // snapshot of auto-detected mask for 'R' reset
    cv::Point drag_start;
    bool is_dragging = false;
    bool left_button = false;  // true = add exclusion; false = erase exclusion
};

static void onMaskEditorMouse(int event, int x, int y, int /*flags*/, void* userdata)
{
    MaskEditorState* s = static_cast<MaskEditorState*>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN) {
        s->drag_start  = cv::Point(x, y);
        s->is_dragging = true;
        s->left_button = true;
    } else if (event == cv::EVENT_RBUTTONDOWN) {
        s->drag_start  = cv::Point(x, y);
        s->is_dragging = true;
        s->left_button = false;
    } else if ((event == cv::EVENT_MOUSEMOVE) && s->is_dragging) {
        // Draw preview
        cv::Mat preview;
        cv::addWeighted(s->first_frame, 0.6,
                        [&]() {
                            cv::Mat overlay(s->first_frame.size(), CV_8UC3, cv::Scalar(0,0,0));
                            cv::Mat mask_vis;
                            cv::cvtColor(s->current_mask, mask_vis, cv::COLOR_GRAY2BGR);
                            // Green where terrain, red where excluded
                            for (int r = 0; r < overlay.rows; ++r)
                                for (int c = 0; c < overlay.cols; ++c)
                                    overlay.at<cv::Vec3b>(r,c) =
                                        (s->current_mask.at<uchar>(r,c) == 255)
                                        ? cv::Vec3b(0,80,0) : cv::Vec3b(0,0,80);
                            return overlay;
                        }(),
                        0.4, 0.0, preview);

        cv::Rect rect(cv::Point(std::min(s->drag_start.x, x), std::min(s->drag_start.y, y)),
                      cv::Point(std::max(s->drag_start.x, x), std::max(s->drag_start.y, y)));
        cv::Scalar rect_color = s->left_button ? cv::Scalar(0,0,255) : cv::Scalar(0,255,0);
        cv::rectangle(preview, rect, rect_color, 2);
        cv::imshow("Mask Editor", preview);

    } else if ((event == cv::EVENT_LBUTTONUP || event == cv::EVENT_RBUTTONUP) && s->is_dragging) {
        s->is_dragging = false;
        cv::Rect rect(cv::Point(std::min(s->drag_start.x, x), std::min(s->drag_start.y, y)),
                      cv::Point(std::max(s->drag_start.x, x), std::max(s->drag_start.y, y)));
        rect = rect & cv::Rect(0, 0, s->current_mask.cols, s->current_mask.rows);
        if (rect.area() > 4) {
            uchar fill = s->left_button ? 0 : 255;   // left=exclude(0), right=restore(255)
            s->current_mask(rect).setTo(fill);
        }
    }
}

// Build a colour overlay image for display
static cv::Mat buildMaskOverlay(const cv::Mat& frame, const cv::Mat& mask)
{
    cv::Mat overlay(frame.size(), CV_8UC3, cv::Scalar(0,0,0));
    for (int r = 0; r < overlay.rows; ++r)
        for (int c = 0; c < overlay.cols; ++c)
            overlay.at<cv::Vec3b>(r,c) =
                (mask.at<uchar>(r,c) == 255)
                ? cv::Vec3b(0,60,0) : cv::Vec3b(0,0,60);

    cv::Mat result;
    cv::addWeighted(frame, 0.65, overlay, 0.35, 0.0, result);
    return result;
}

// ---------------------------------------------------------------------------
// Altitude measurement callback
// ---------------------------------------------------------------------------
struct AltitudeMeasureState {
    cv::Point pt1, pt2;
    bool drawing = false;
    bool done    = false;
    cv::Mat frame;
};

static void onAltitudeMouseDraw(int event, int x, int y, int /*flags*/, void* userdata)
{
    AltitudeMeasureState* d = static_cast<AltitudeMeasureState*>(userdata);
    if (event == cv::EVENT_LBUTTONDOWN) {
        d->pt1 = d->pt2 = cv::Point(x, y);
        d->drawing = true;
        d->done    = false;
    } else if (event == cv::EVENT_MOUSEMOVE && d->drawing) {
        d->pt2 = cv::Point(x, y);
        cv::Mat tmp = d->frame.clone();
        cv::line(tmp, d->pt1, d->pt2, cv::Scalar(0,255,0), 2);
        cv::imshow("Altitude Estimation", tmp);
    } else if (event == cv::EVENT_LBUTTONUP) {
        d->drawing = false;
        d->done    = true;
        cv::Mat tmp = d->frame.clone();
        cv::line(tmp, d->pt1, d->pt2, cv::Scalar(0,255,0), 2);
        cv::imshow("Altitude Estimation", tmp);
    }
}

// ---------------------------------------------------------------------------
// VideoPreprocessor
// ---------------------------------------------------------------------------

VideoPreprocessor::VideoPreprocessor(const std::string& input_video_path,
                                     const std::string& output_video_path,
                                     const std::string& masks_dir)
    : input_path_(input_video_path)
    , output_path_(output_video_path)
    , masks_dir_(masks_dir)
{
    camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
    dist_coeffs_   = cv::Mat::zeros(1, 5, CV_64F);
    mkdir(masks_dir_.c_str(), 0755);
}

VideoMetadata VideoPreprocessor::process(int frame_skip)
{
    VideoMetadata metadata;
    cv::VideoCapture cap(input_path_);
    if (!cap.isOpened()) {
        std::cerr << "Error: could not open video: " << input_path_ << std::endl;
        return metadata;
    }

    metadata.fps          = cap.get(cv::CAP_PROP_FPS);
    metadata.total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    metadata.width        = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    metadata.height       = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    metadata.processed_video_path = output_path_;

    std::cout << "Video: " << metadata.width << "x" << metadata.height
              << " @ " << metadata.fps << " FPS, "
              << metadata.total_frames << " frames" << std::endl;

    // -------------------------------------------------------------------------
    // Step 1: Read first frame for interactive steps
    // -------------------------------------------------------------------------
    cv::Mat first_frame;
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    if (!cap.read(first_frame)) {
        std::cerr << "Error: could not read first frame." << std::endl;
        return metadata;
    }

    // -------------------------------------------------------------------------
    // Step 2: Build or load feature mask
    // -------------------------------------------------------------------------
    // Derive a safe filename stem from the video path
    std::string stem = input_path_;
    for (char& c : stem) if (c == '/' || c == '\\' || c == ' ') c = '_';
    std::string mask_path = masks_dir_ + "/" + stem + "_mask.png";

    cv::Mat feature_mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
    if (!feature_mask.empty() &&
        feature_mask.size() == cv::Size(metadata.width, metadata.height)) {
        std::cout << "Loaded cached mask: " << mask_path << std::endl;
    } else {
        // Auto-detect static overlays
        std::cout << "Analysing temporal variance to detect static overlays..." << std::endl;
        cv::Mat auto_mask = generateStaticOverlayMask(cap, 60);

        // Interactive refinement
        feature_mask = interactiveRefineOverlayMask(first_frame, auto_mask, mask_path);
    }
    metadata.feature_mask = feature_mask;

    // -------------------------------------------------------------------------
    // Step 3: Estimate camera angle
    // -------------------------------------------------------------------------
    metadata.estimated_mean_angle = estimateCameraAngle(first_frame);
    if (metadata.estimated_mean_angle == -1.0)
        std::cout << "Camera angle: nadir (or unable to detect horizon)" << std::endl;
    else
        std::cout << "Estimated camera pitch angle: " << metadata.estimated_mean_angle << " deg" << std::endl;

    // -------------------------------------------------------------------------
    // Step 4: Estimate altitude (cached like the mask, so iterative test runs
    // don't require re-answering the interactive line-draw prompt every time).
    // Delete the cache file to force a fresh estimate.
    // -------------------------------------------------------------------------
    std::string altitude_cache_path = masks_dir_ + "/" + stem + "_altitude.txt";
    std::ifstream altitude_in(altitude_cache_path);
    if (altitude_in.good() && (altitude_in >> metadata.estimated_mean_altitude)) {
        std::cout << "Loaded cached altitude estimate: "
                  << metadata.estimated_mean_altitude << " m ("
                  << altitude_cache_path << ")" << std::endl;
    } else {
        metadata.estimated_mean_altitude = estimateAltitudeFromReference(cap, metadata.total_frames, CameraModel::kDefaultHfovDeg);
        if (metadata.estimated_mean_altitude > 0) {
            std::ofstream altitude_out(altitude_cache_path);
            altitude_out << metadata.estimated_mean_altitude;
            std::cout << "Altitude estimate cached: " << altitude_cache_path << std::endl;
        }
    }

    // -------------------------------------------------------------------------
    // Step 5: Write CLAHE-enhanced output video
    //   The mask is NOT used to zero pixels — it is only returned so the
    //   calling pipeline can pass it to feature extractors.
    // -------------------------------------------------------------------------
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    cv::VideoWriter writer(output_path_,
                           cv::VideoWriter::fourcc('M','J','P','G'),
                           metadata.fps / frame_skip,
                           cv::Size(metadata.width, metadata.height), true);

    cv::Mat frame, gray, enhanced;
    int idx = 0;
    while (cap.read(frame)) {
        if (idx % frame_skip != 0) { ++idx; continue; }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(gray, gray);
        cv::cvtColor(gray, enhanced, cv::COLOR_GRAY2BGR);

        writer.write(enhanced);
        ++idx;

        if (idx % 200 == 0)
            std::cout << "  Writing frame " << idx << " / " << metadata.total_frames << std::endl;
    }

    cap.release();
    writer.release();
    std::cout << "Preprocessed video saved: " << output_path_ << std::endl;
    return metadata;
}

// ---------------------------------------------------------------------------
// generateStaticOverlayMask
// ---------------------------------------------------------------------------
cv::Mat VideoPreprocessor::generateStaticOverlayMask(cv::VideoCapture& cap,
                                                      int sample_frames)
{
    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    cv::Mat sum   = cv::Mat::zeros(h, w, CV_32FC1);
    cv::Mat sqsum = cv::Mat::zeros(h, w, CV_32FC1);

    cv::Mat frame, gray;
    int count = 0;
    for (int i = 0; i < sample_frames; ++i) {
        int target = i * (total / sample_frames);
        cap.set(cv::CAP_PROP_POS_FRAMES, target);
        if (!cap.read(frame)) break;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat f32;
        gray.convertTo(f32, CV_32FC1);
        cv::accumulate(f32, sum);
        cv::accumulateSquare(f32, sqsum);
        ++count;
    }

    if (count == 0) return cv::Mat::ones(h, w, CV_8U) * 255;

    cv::Mat mean   = sum   / count;
    cv::Mat sqmean = sqsum / count;
    cv::Mat variance = sqmean - mean.mul(mean);

    cv::Mat mask;
    cv::threshold(variance, mask, 10.0, 255, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    int static_px = cv::countNonZero(~mask);
    std::cout << "Auto-detected " << static_px << " static-overlay pixels." << std::endl;
    return mask;
}

// ---------------------------------------------------------------------------
// interactiveRefineOverlayMask
// ---------------------------------------------------------------------------
cv::Mat VideoPreprocessor::interactiveRefineOverlayMask(const cv::Mat& first_frame,
                                                         const cv::Mat& auto_mask,
                                                         const std::string& mask_path)
{
    MaskEditorState state;
    state.first_frame  = first_frame.clone();
    state.auto_mask    = auto_mask.clone();
    state.current_mask = auto_mask.clone();

    std::cout << "\n=== MASK EDITOR ===" << std::endl;
    std::cout << "  Green tint = valid terrain (features will be extracted here)" << std::endl;
    std::cout << "  Red tint   = excluded region (logos, subtitles)" << std::endl;
    std::cout << "  LEFT-DRAG  = add exclusion zone" << std::endl;
    std::cout << "  RIGHT-DRAG = restore terrain region" << std::endl;
    std::cout << "  'S'        = save and continue" << std::endl;
    std::cout << "  'R'        = reset to auto-detected mask" << std::endl;

    cv::namedWindow("Mask Editor", cv::WINDOW_NORMAL);
    cv::resizeWindow("Mask Editor", 1200, 900);
    cv::setMouseCallback("Mask Editor", onMaskEditorMouse, &state);
    cv::imshow("Mask Editor", buildMaskOverlay(first_frame, state.current_mask));

    while (true) {
        int key = cv::waitKey(30) & 0xFF;
        if (key == 's' || key == 'S') break;
        if (key == 'r' || key == 'R') {
            state.current_mask = state.auto_mask.clone();
            cv::imshow("Mask Editor", buildMaskOverlay(first_frame, state.current_mask));
        }
        // Refresh display after each drag completion
        if (!state.is_dragging)
            cv::imshow("Mask Editor", buildMaskOverlay(first_frame, state.current_mask));
    }
    cv::destroyWindow("Mask Editor");

    cv::imwrite(mask_path, state.current_mask);
    std::cout << "Mask saved: " << mask_path << std::endl;
    return state.current_mask;
}

// ---------------------------------------------------------------------------
// estimateCameraAngle
// ---------------------------------------------------------------------------
double VideoPreprocessor::estimateCameraAngle(const cv::Mat& frame)
{
    cv::Mat gray, edges;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150);

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges, lines, 1, CV_PI / 180, 150);

    for (const auto& line : lines) {
        float theta = line[1];
        // Mostly-horizontal lines indicate a horizon
        if (theta > CV_PI/2 - 0.2f && theta < CV_PI/2 + 0.2f) {
            double rho      = line[0];
            double center_y = frame.rows / 2.0;
            double vfov_deg = 60.0;
            return ((center_y - rho) / frame.rows) * vfov_deg;
        }
    }
    return -1.0;  // nadir or no horizon detected
}

// ---------------------------------------------------------------------------
// estimateAltitudeFromReference
// ---------------------------------------------------------------------------
double VideoPreprocessor::estimateAltitudeFromReference(cv::VideoCapture& cap,
                                                         int total_frames,
                                                         double camera_hfov_deg)
{
    std::cout << "\n=== ALTITUDE ESTIMATION ===" << std::endl;
    std::cout << "Find a frame with a known-size object visible (car ~4.5m, road lane ~3.5m," << std::endl;
    std::cout << "shipping container, etc.) -- frame 0 isn't always usable (e.g. footage" << std::endl;
    std::cout << "that opens over open water)." << std::endl;
    std::cout << "  Trackbar, or 'a'/'d' = +/-1 frame, 'A'/'D' = +/-10 frames, to scrub." << std::endl;
    std::cout << "  SPACE = lock this frame and measure | 'S' = skip altitude estimation" << std::endl;

    cv::namedWindow("Altitude Estimation", cv::WINDOW_NORMAL);
    cv::resizeWindow("Altitude Estimation", 1200, 900);

    int frame_idx = 0;
    cv::Mat frame;
    auto seek = [&](int idx) {
        frame_idx = std::max(0, std::min(total_frames - 1, idx));
        cap.set(cv::CAP_PROP_POS_FRAMES, frame_idx);
        if (!cap.read(frame)) return;
        cv::Mat disp = frame.clone();
        cv::putText(disp, "Frame " + std::to_string(frame_idx) + "/" + std::to_string(total_frames - 1),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::imshow("Altitude Estimation", disp);
    };

    int trackbar_pos = 0;
    std::function<void(int)> onTrackbar = seek;
    cv::createTrackbar("Frame", "Altitude Estimation", &trackbar_pos,
                       std::max(0, total_frames - 1),
                       [](int pos, void* userdata) {
                           auto* fn = static_cast<std::function<void(int)>*>(userdata);
                           (*fn)(pos);
                       }, &onTrackbar);

    seek(0);

    bool locked = false;
    while (!locked) {
        int key = cv::waitKey(30) & 0xFF;
        if (key == 's' || key == 'S') {
            cv::destroyWindow("Altitude Estimation");
            return -1.0;
        }
        if (key == ' ') { locked = true; break; }

        int step = 0;
        if      (key == 'd') step = 1;
        else if (key == 'a') step = -1;
        else if (key == 'D') step = 10;
        else if (key == 'A') step = -10;
        if (step != 0) {
            trackbar_pos = std::max(0, std::min(total_frames - 1, frame_idx + step));
            cv::setTrackbarPos("Frame", "Altitude Estimation", trackbar_pos);
        }
    }

    std::cout << "Locked frame " << frame_idx << ". Draw a line over the known-size object." << std::endl;
    std::cout << "'SPACE' when done drawing | 'S' to skip" << std::endl;

    AltitudeMeasureState data;
    data.frame = frame.clone();
    cv::imshow("Altitude Estimation", data.frame);
    cv::setMouseCallback("Altitude Estimation", onAltitudeMouseDraw, &data);

    while (true) {
        int key = cv::waitKey(10) & 0xFF;
        if (key == 's' || key == 'S') {
            cv::destroyWindow("Altitude Estimation");
            return -1.0;
        }
        if (data.done && key == ' ') break;
    }
    cv::destroyWindow("Altitude Estimation");

    double pixel_len = cv::norm(data.pt1 - data.pt2);
    if (pixel_len < 5.0) {
        std::cerr << "Line too short — skipping altitude estimation." << std::endl;
        return -1.0;
    }

    double real_world_m;
    std::cout << "Enter real-world length of the object in metres (e.g. 4.5): ";
    std::cin >> real_world_m;

    double gsd               = real_world_m / pixel_len;
    double footprint_width_m = gsd * frame.cols;
    double hfov_rad          = camera_hfov_deg * M_PI / 180.0;
    double altitude          = (footprint_width_m / 2.0) / std::tan(hfov_rad / 2.0);

    std::cout << "Estimated altitude: ~" << altitude << " m" << std::endl;
    return altitude;
}
