#include "VideoProcessing.hpp"
#include "CoordinateUtils.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>

std::vector<ReferenceCrop> generateReferenceCropsGrid(
    const cv::Mat &map,
    double center_lat, double center_lng,
    int center_x, int center_y,
    double mpp,
    double meters_per_degree_lat,
    double meters_per_degree_lng,
    int grid_spacing_meters,
    int crop_size_meters)
{
    std::vector<ReferenceCrop> crops;

    // Calculate crop size in pixels
    int crop_size_px = static_cast<int>(std::round(crop_size_meters / mpp));

    // Calculate how many meters the map spans
    double map_width_meters = map.cols * mpp;
    double map_height_meters = map.rows * mpp;

    // Calculate the geographic bounds of the map
    double half_width_deg_lat = (map_width_meters / 2.0) / meters_per_degree_lat;
    double half_height_deg_lng = (map_height_meters / 2.0) / meters_per_degree_lng;

    double min_lat = center_lat - half_height_deg_lng;
    double max_lat = center_lat + half_height_deg_lng;
    double min_lng = center_lng - half_width_deg_lat;
    double max_lng = center_lng + half_width_deg_lat;

    // Calculate grid spacing in degrees
    double grid_spacing_lat = grid_spacing_meters / meters_per_degree_lat;
    double grid_spacing_lng = grid_spacing_meters / meters_per_degree_lng;

    std::cout << "\n📐 Generating reference crop grid:" << std::endl;
    std::cout << "   Grid spacing: " << grid_spacing_meters << "m" << std::endl;
    std::cout << "   Crop size: " << crop_size_meters << "m (" << crop_size_px << "px)" << std::endl;
    std::cout << "   Map coverage: " << std::fixed << std::setprecision(1)
              << map_width_meters << "m × " << map_height_meters << "m" << std::endl;

    int crop_count = 0;

    // Generate grid of crops
    for (double lat = min_lat; lat <= max_lat; lat += grid_spacing_lat)
    {
        for (double lng = min_lng; lng <= max_lng; lng += grid_spacing_lng)
        {
            // Convert lat/lng to pixel coordinates
            cv::Point pixel = CoordinateUtils::latLngToPixel(
                lat, lng, center_lat, center_lng,
                center_x, center_y, mpp,
                meters_per_degree_lat, meters_per_degree_lng);

            // Define crop rectangle
            cv::Rect crop_rect(
                pixel.x - crop_size_px / 2,
                pixel.y - crop_size_px / 2,
                crop_size_px,
                crop_size_px);

            // Ensure crop is within map bounds
            crop_rect = crop_rect & cv::Rect(0, 0, map.cols, map.rows);

            // Only add if crop has sufficient size
            if (crop_rect.width > crop_size_px * 0.5 &&
                crop_rect.height > crop_size_px * 0.5)
            {
                cv::Mat cropped = map(crop_rect).clone();
                crops.push_back(ReferenceCrop({lat, lng}, cropped));
                crop_count++;
            }
        }
    }

    std::cout << "   ✓ Generated " << crop_count << " reference crops" << std::endl;

    return crops;
}

void processVideoNavigation(
    const std::string &video_path,
    const cv::Mat &reference_map,
    const std::vector<ReferenceCrop> &reference_crops,
    double center_lat, double center_lng,
    int center_x, int center_y,
    double mpp,
    double meters_per_degree_lat,
    double meters_per_degree_lng,
    PositionAlgorithm algorithm,
    const std::string &location_name,
    const std::string &video_name,
    int frame_skip,
    std::pair<double, double> initial_position,
    int start_frame_idx)
{
    // ==================== INITIALIZATION ====================

    std::cout << "\n"
              << std::string(80, '=') << std::endl;
    std::cout << "🎥 VIDEO-BASED NAVIGATION" << std::endl;
    std::cout << "   Location: " << location_name << std::endl;
    std::cout << "   Video: " << video_name << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Open video file
    cv::VideoCapture video(video_path);
    if (!video.isOpened())
    {
        std::cerr << "❌ Error: Could not open video file: " << video_path << std::endl;
        return;
    }

    // Get video properties
    int total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = video.get(cv::CAP_PROP_FPS);
    int video_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
    int video_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "\n📊 Video Properties:" << std::endl;
    std::cout << "   Resolution: " << video_width << "×" << video_height << std::endl;
    std::cout << "   Total frames: " << total_frames << std::endl;
    std::cout << "   FPS: " << fps << std::endl;
    std::cout << "   Duration: " << std::fixed << std::setprecision(1)
              << (total_frames / fps) << " seconds" << std::endl;
    std::cout << "   Frame skip: " << frame_skip << " (processing every "
              << frame_skip << (frame_skip == 1 ? " frame)" : " frames)") << std::endl;

    // Create position estimator
    std::unique_ptr<IPositionEstimator> estimator = createPositionEstimator(algorithm);
    std::string algorithm_name = estimator->getName();
    std::cout << "   Algorithm: " << algorithm_name << " + Kalman Filter" << std::endl;
    std::cout << "   Reference crops: " << reference_crops.size() << std::endl;

    // Initialize Kalman filter
    // Start with higher measurement noise since we don't have initial heading/speed
    DroneKalmanFilter kalman_filter(1.0, 20.0);
    bool kalman_initialized = false;

    if (initial_position.first != 0.0 && initial_position.second != 0.0)
    {
        kalman_filter.initialize(initial_position.first, initial_position.second, 0.0, 0.0);
        kalman_initialized = true;
        std::cout << "   Initial position: (" << initial_position.first
                  << ", " << initial_position.second << ")" << std::endl;
    }
    else
    {
        std::cout << "   ⚠️  No initial position provided - will estimate from first frame" << std::endl;
    }

    // Setup visualization windows
    cv::namedWindow("Video Navigation", cv::WINDOW_NORMAL);
    cv::namedWindow("Video Frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Navigation", 1200, 800);

    // Setup output video writer
    mkdir("Videos", 0755);
    std::string output_video_path = "Videos/navigation_" + algorithm_name + "_" +
                                    location_name + "_" + video_name + ".avi";

    cv::VideoWriter video_writer;
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    video_writer.open(output_video_path, fourcc, fps / frame_skip,
                      cv::Size(1200, 800), true);

    if (video_writer.isOpened())
    {
        std::cout << "   ✓ Recording output: " << output_video_path << std::endl;
    }
    else
    {
        std::cerr << "   ⚠️  Video recording disabled" << std::endl;
    }

    // Setup telemetry CSV
    mkdir("CSV Files", 0755);
    std::string telemetry_path = "CSV Files/video_telemetry_" + algorithm_name + "_" +
                                 location_name + "_" + video_name + ".csv";
    std::ofstream telemetry_file(telemetry_path);

    if (!telemetry_file.is_open())
    {
        std::cerr << "❌ Error: Could not create telemetry file" << std::endl;
        return;
    }

    telemetry_file << "Frame,Time_Sec,Raw_Lat,Raw_Lng,Predicted_Lat,Predicted_Lng,"
                   << "Filtered_Lat,Filtered_Lng,Match_Confidence,Innovation_M,"
                   << "Outlier_Rejected,Best_Match_Idx,Algorithm\n";

    // Storage for trajectory
    std::vector<std::pair<double, double>> raw_positions;
    std::vector<std::pair<double, double>> filtered_positions;
    std::vector<std::pair<double, double>> predicted_positions;
    std::vector<double> confidences;

    int outliers_rejected = 0;
    int low_confidence_count = 0;

    // ==================== MAIN PROCESSING LOOP ====================

    std::cout << "\n🚁 Processing video frames..." << std::endl;

    cv::Mat frame;
    int frame_idx = 0;
    int processed_frames = 0;

    // Expected crop size for drone view matching
    int expected_crop_size = static_cast<int>(std::round(100.0 / mpp));

    while (video.read(frame))
    {
        // Skip frames if needed

        if (frame_idx < start_frame_idx) {
            frame_idx++;
            continue;
        }
        
        if (frame_idx % frame_skip != 0)
        {
            frame_idx++;
            continue;
        }

        double time_sec = frame_idx / fps;
        double dt = frame_skip / fps; // Time delta between processed frames

        // Preprocess frame - resize to expected crop size if needed
        cv::Mat processed_frame;

        // 1. The video is 1920x1080. The HUD graphics are mostly on the left and center-bottom.
        // Let's grab a clean 600x600 square from the center-right of the frame to avoid the red missile/text.
        // You may need to tweak these offsets based on exactly where the clean video is.
        int safe_width = 600;
        int safe_height = 600;
        int start_x = (frame.cols / 2) - 100; // Shifted slightly right to avoid left-side text
        int start_y = (frame.rows / 2) - (safe_height / 2);

        cv::Rect clean_roi(start_x, start_y, safe_width, safe_height);

        // Ensure ROI is within bounds
        clean_roi = clean_roi & cv::Rect(0, 0, frame.cols, frame.rows);

        if (clean_roi.width > 0 && clean_roi.height > 0)
        {
            cv::Mat cropped = frame(clean_roi);

            // 2. Resize it to a standard CV processing size (e.g., 250x250 or 300x300)
            // This gives ORB enough pixels to find corners, but isn't so massive it slows down CPU
            cv::resize(cropped, processed_frame, cv::Size(300, 300));
        }
        else
        {
            processed_frame = frame.clone();
        }
        cv::imshow("Debug - Processed Frame", processed_frame);
        cv::waitKey(1); // For visualization
        // ==================== KALMAN PREDICTION ====================

        std::pair<double, double> predicted_position = {0.0, 0.0};
        if (kalman_initialized)
        {
            predicted_position = kalman_filter.predict(dt);
            predicted_positions.push_back(predicted_position);
        }

        // ==================== VISUAL ESTIMATION & SPATIAL FILTERING ====================

        // 1. Figure out our last known position (Keep your original logic)
        std::pair<double, double> last_position = filtered_positions.empty() ? initial_position : filtered_positions.back();
        if (last_position.first == 0.0 && last_position.second == 0.0)
        {
            last_position = {center_lat, center_lng}; // Use map center as fallback
        }

        // 2. Define where we want to center our search radius
        std::pair<double, double> search_center = kalman_initialized ? predicted_position : last_position;

        // 3. Filter the crops
        std::vector<ReferenceCrop> local_crops;

        // If we truly have no idea where we are, search the whole map
        if (search_center.first == 0.0 && search_center.second == 0.0)
        {
            local_crops = reference_crops;
        }
        else
        {
            // Local Search: Only grab crops within a 400-meter radius
            const double search_radius_meters = 400.0;

            for (const auto &crop : reference_crops)
            {
                // Simple Euclidean distance converted to meters
                double d_lat = (crop.coordinates.first - search_center.first) * meters_per_degree_lat;
                double d_lng = (crop.coordinates.second - search_center.second) * meters_per_degree_lng;
                double distance = std::sqrt(d_lat * d_lat + d_lng * d_lng);

                if (distance <= search_radius_meters)
                {
                    local_crops.push_back(crop);
                }
            }

            // Safety net: if the drone flies off the mapped area, revert to global search
            if (local_crops.empty())
            {
                local_crops = reference_crops;
            }
        }

        // 4. Pass the tiny subset of local_crops to the estimator
        PositionEstimate estimate = estimator->estimatePosition(
            processed_frame, local_crops, last_position);

        std::pair<double, double> raw_position = estimate.position;
        double confidence = estimate.confidence;
        int best_match_idx = estimate.best_match_idx;

        raw_positions.push_back(raw_position);
        confidences.push_back(confidence);

        // ==================== KALMAN UPDATE ====================

        std::pair<double, double> filtered_position;
        double innovation = 0.0;
        bool outlier_rejected = false;

        if (!kalman_initialized)
        {
            // Initialize Kalman filter with first estimate
            kalman_filter.initialize(raw_position.first, raw_position.second, 0.0, 0.0);
            kalman_initialized = true;
            filtered_position = raw_position;
            std::cout << "   ✓ Kalman filter initialized at frame " << frame_idx
                      << " with position (" << raw_position.first << ", "
                      << raw_position.second << ")" << std::endl;
        }
        else
        {
            // Calculate innovation
            innovation = kalman_filter.getInnovation(raw_position.first, raw_position.second);

            // Adaptive innovation threshold
            double innovation_threshold = confidence > 0.7 ? 80.0 : 50.0;

            // Update Kalman filter
            filtered_position = kalman_filter.update(
                raw_position.first, raw_position.second,
                confidence, innovation_threshold);

            // Check if outlier was rejected
            if (innovation > innovation_threshold && confidence < 0.7)
            {
                outlier_rejected = true;
                outliers_rejected++;
            }
        }

        filtered_positions.push_back(filtered_position);

        if (confidence < 0.5)
        {
            low_confidence_count++;
        }

        // ==================== TELEMETRY LOGGING ====================

        telemetry_file << frame_idx << ","
                       << std::fixed << std::setprecision(2) << time_sec << ","
                       << std::setprecision(8)
                       << raw_position.first << "," << raw_position.second << ","
                       << predicted_position.first << "," << predicted_position.second << ","
                       << filtered_position.first << "," << filtered_position.second << ","
                       << std::setprecision(4) << confidence << ","
                       << std::setprecision(2) << innovation << ","
                       << (outlier_rejected ? "1" : "0") << ","
                       << best_match_idx << ","
                       << algorithm_name << "\n";

        // ==================== VISUALIZATION ====================

        // Create visualization on reference map
        cv::Mat display_map = reference_map.clone();

        // Draw all filtered positions (trajectory)
        for (size_t i = 1; i < filtered_positions.size(); i++)
        {
            cv::Point pt1 = CoordinateUtils::latLngToPixel(
                filtered_positions[i - 1].first, filtered_positions[i - 1].second,
                center_lat, center_lng, center_x, center_y, mpp,
                meters_per_degree_lat, meters_per_degree_lng);

            cv::Point pt2 = CoordinateUtils::latLngToPixel(
                filtered_positions[i].first, filtered_positions[i].second,
                center_lat, center_lng, center_x, center_y, mpp,
                meters_per_degree_lat, meters_per_degree_lng);

            // Color-code by confidence
            double conf = confidences[i];
            cv::Scalar color = conf > 0.7 ? cv::Scalar(0, 255, 0) : // Green: high conf
                                   conf > 0.5 ? cv::Scalar(0, 255, 255)
                                              :           // Yellow: medium
                                   cv::Scalar(0, 0, 255); // Red: low

            cv::line(display_map, pt1, pt2, color, 2);
        }

        // Draw current position (larger marker)
        if (!filtered_positions.empty())
        {
            cv::Point current_pt = CoordinateUtils::latLngToPixel(
                filtered_position.first, filtered_position.second,
                center_lat, center_lng, center_x, center_y, mpp,
                meters_per_degree_lat, meters_per_degree_lng);

            cv::circle(display_map, current_pt, 8, cv::Scalar(255, 0, 255), -1);   // Magenta
            cv::circle(display_map, current_pt, 10, cv::Scalar(255, 255, 255), 2); // White outline
        }

        // Add info overlay
        std::ostringstream info;
        info << "Frame: " << frame_idx << "/" << total_frames
             << " | Time: " << std::fixed << std::setprecision(1) << time_sec << "s"
             << " | Conf: " << std::setprecision(3) << confidence;

        cv::putText(display_map, info.str(), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        std::ostringstream pos_info;
        pos_info << "Position: " << std::fixed << std::setprecision(6)
                 << filtered_position.first << ", " << filtered_position.second;
        cv::putText(display_map, pos_info.str(), cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        if (outlier_rejected)
        {
            cv::putText(display_map, "OUTLIER REJECTED", cv::Point(10, 90),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }

        // Show windows
        cv::imshow("Video Navigation", display_map);
        cv::imshow("Video Frame", frame);

        // Write to output video
        if (video_writer.isOpened())
        {
            cv::Mat output_frame;
            cv::resize(display_map, output_frame, cv::Size(1200, 800));
            video_writer.write(output_frame);
        }

        // Progress update
        if (processed_frames % 10 == 0 || processed_frames == 0)
        {
            std::cout << "   Frame " << frame_idx << "/" << total_frames
                      << " (" << std::setprecision(1) << (frame_idx * 100.0 / total_frames)
                      << "%) - Conf: " << std::setprecision(3) << confidence << std::endl;
        }

        // User can press 'q' to quit early, space to pause
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27)
        { // q or ESC
            std::cout << "\n⚠️  Processing cancelled by user" << std::endl;
            break;
        }
        else if (key == ' ')
        { // Space to pause
            std::cout << "   ⏸️  Paused - press any key to continue..." << std::endl;
            cv::waitKey(0);
        }

        frame_idx++;
        processed_frames++;
    }

    // ==================== CLEANUP & SUMMARY ====================

    video.release();
    if (video_writer.isOpened())
    {
        video_writer.release();
    }
    telemetry_file.close();

    std::cout << "\n"
              << std::string(80, '=') << std::endl;
    std::cout << "✅ VIDEO PROCESSING COMPLETE" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "📊 Statistics:" << std::endl;
    std::cout << "   Processed frames: " << processed_frames << std::endl;
    std::cout << "   Outliers rejected: " << outliers_rejected
              << " (" << std::setprecision(1)
              << (100.0 * outliers_rejected / processed_frames) << "%)" << std::endl;
    std::cout << "   Low confidence (<0.5): " << low_confidence_count
              << " (" << (100.0 * low_confidence_count / processed_frames) << "%)" << std::endl;

    if (!confidences.empty())
    {
        double avg_conf = 0.0;
        for (double c : confidences)
            avg_conf += c;
        avg_conf /= confidences.size();
        std::cout << "   Average confidence: " << std::setprecision(3) << avg_conf << std::endl;
    }

    std::cout << "\n📁 Output files:" << std::endl;
    std::cout << "   Video: " << output_video_path << std::endl;
    std::cout << "   Telemetry: " << telemetry_path << std::endl;

    std::cout << "\nPress any key in the visualization window to close..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}
