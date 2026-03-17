#include "GlobalLocator.hpp"
#include <iostream>

// Struct to pass data to the pthread safely
struct LocatorThreadData {
    int start_idx;
    int end_idx;
    const std::vector<ReferenceCrop>* crops;
    cv::Mat target_descriptors;
    int best_idx;
    float best_score;
};

// pthread routine

void* locatorThreadFunc(void* arg)
{
    LocatorThreadData* data = static_cast<LocatorThreadData*>(arg);
    data->best_idx = -1;
    data->best_score = -1.0f;
    
    if (data->target_descriptors.empty()) return nullptr;

    // CRITICAL: OpenCV feature extractors are NOT thread-safe.
    // Each thread MUST create its own local ORB detector and matcher.
    cv::Ptr<cv::ORB> local_orb = cv::ORB::create(500);
    cv::BFMatcher local_matcher(cv::NORM_HAMMING);

    for (int i = data->start_idx; i < data->end_idx; ++i)
    {
        const ReferenceCrop& crop = (*data->crops)[i];
        if (crop.image.empty()) continue;

        // Extract features from the reference map crop
        std::vector<cv::KeyPoint> kp_r;
        cv::Mat desc_ref;
        local_orb->detectAndCompute(crop.image, cv::noArray(), kp_r, desc_ref);

        if (desc_ref.empty() || kp_r.size() < 10) continue;

        std::vector<std::vector<cv::DMatch>> knn_matches;
        try 
        {
            // Read-only access to data->target_descriptors is thread-safe
            local_matcher.knnMatch(data->target_descriptors, desc_ref, knn_matches, 2);
        
        } catch (...) {continue;}

        // Apply Lowe's ratio test
        std::vector<cv::DMatch> good_matches;
        for (const auto& m : knn_matches)
        {
            if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance)
            {
                good_matches.push_back(m[0]);
            }
        }

        // Calculate score
        float match_score = 0.0f;
        if (!good_matches.empty())
        {
            float avg_distance = 0.0f;
            for (const auto& m : good_matches)
            {
                avg_distance += m.distance;
            }
            avg_distance /= good_matches.size();
            match_score = static_cast<float>(good_matches.size()) / (avg_distance / 256.0f + 1.0f);
        }

        // Update local best match
        if (match_score > data->best_score)
        {
            data->best_score = match_score;
            data->best_idx = i;
        }
    }
    return nullptr;
}

#include "GlobalLocator.hpp"
#include <iostream>

// ... (Keep the LocatorThreadData and locatorThreadFunc exactly as they are) ...

InitializationData GlobalLocator::findStartingPosition(
    const std::string& video_path,
    const std::vector<ReferenceCrop>& crops,
    int num_threads,
    int frame_skip)
{
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "❌ Could not open video for initialization." << std::endl;
        return {false, 0, {0.0, 0.0}};
    }

    cv::Mat frame;
    int current_frame_idx = 0;
    
    // We create a persistent ORB detector for the target frames
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);

    // Loop through the video, searching for a clear frame
    while (cap.read(frame)) {
        
        // Only check every Nth frame to speed up the search
        if (current_frame_idx % frame_skip != 0) {
            current_frame_idx++;
            continue;
        }

        std::cout << "🔍 Global Search: Analyzing Frame " << current_frame_idx << "..." << std::endl;

        // Preprocess to remove HUD
        cv::Mat processed_frame;
        int crop_dim = 600;
        int x_offset = (frame.cols / 2) - 50; 
        int y_offset = (frame.rows - crop_dim) / 2;
        cv::Rect clean_roi(x_offset, y_offset, crop_dim, crop_dim);
        clean_roi = clean_roi & cv::Rect(0, 0, frame.cols, frame.rows);
        
        if (clean_roi.width > 0 && clean_roi.height > 0) {
            // Contrast boost for hazy environments
            cv::Mat cropped = frame(clean_roi);
            cv::Mat contrast_boosted;
            cropped.convertTo(contrast_boosted, -1, 1.2, 10); 
            cv::resize(contrast_boosted, processed_frame, cv::Size(300, 300));
        } else {
            processed_frame = frame.clone();
        }

        // Try to extract features
        std::vector<cv::KeyPoint> target_kp;
        cv::Mat target_desc;
        orb->detectAndCompute(processed_frame, cv::noArray(), target_kp, target_desc);

        // THE FIX: If it's a blurry water frame, it will fail this check and continue to the next frame!
        if (target_desc.empty() || target_kp.size() < 30) {
            std::cout << "   ⚠️ Frame " << current_frame_idx << " rejected (Insufficient features - likely water/haze). Scanning forward..." << std::endl;
            current_frame_idx++;
            continue; 
        }

        std::cout << "   ✅ Valid frame found. Spawning " << num_threads << " POSIX threads..." << std::endl;

        // --- Execute the Multithreaded Search on this valid frame ---
        std::vector<pthread_t> threads(num_threads);
        std::vector<LocatorThreadData> thread_data(num_threads);
        int chunk_size = crops.size() / num_threads;
        
        for (int i = 0; i < num_threads; ++i) {
            thread_data[i].start_idx = i * chunk_size;
            thread_data[i].end_idx = (i == num_threads - 1) ? crops.size() : (i + 1) * chunk_size;
            thread_data[i].crops = &crops;
            thread_data[i].target_descriptors = target_desc; 
            pthread_create(&threads[i], nullptr, locatorThreadFunc, &thread_data[i]);
        }

        int best_global_idx = -1;
        float best_global_score = -1.0f;

        for (int i = 0; i < num_threads; ++i) {
            pthread_join(threads[i], nullptr);
            if (thread_data[i].best_score > best_global_score) {
                best_global_score = thread_data[i].best_score;
                best_global_idx = thread_data[i].best_idx;
            }
        }
        // -----------------------------------------------------------

        // If we found a strong match (tune this threshold if needed, e.g., > 20.0f)
        if (best_global_idx != -1 && best_global_score > 15.0f) {
            std::cout << "🎯 Initialization Locked at Frame " << current_frame_idx 
                      << "! Confidence score: " << best_global_score << std::endl;
            cap.release();
            return {true, current_frame_idx, crops[best_global_idx].coordinates};
        } else {
             std::cout << "   ⚠️ Frame " << current_frame_idx << " features extracted, but no strong map match found. Scanning forward..." << std::endl;
        }

        current_frame_idx++;
    }

    cap.release();
    std::cerr << "❌ CRITICAL FAILURE: Could not find any valid initialization frame in the entire video." << std::endl;
    return {false, 0, {0.0, 0.0}};
}
