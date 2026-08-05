#include "SelfReferentialExperiment.hpp"
#include "TelemetryImporter.hpp"
#include "GroundTruthAnnotator.hpp"
#include "ORBFeatureEstimator.hpp"
#include "EdgeProcessor.hpp"
#include "CoordinateUtils.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <sys/stat.h>

namespace {

constexpr double kMetersPerDegLat = 111320.0;

double metersPerDegLng(double lat_deg) {
    return kMetersPerDegLat * std::cos(lat_deg * M_PI / 180.0);
}

std::string frameImagePath(const DatasetSampleConfig& cfg, int frame_num_1based) {
    std::ostringstream oss;
    oss << cfg.image_dir << "/" << cfg.image_prefix << "_"
        << std::setw(4) << std::setfill('0') << frame_num_1based << ".JPG";
    return oss.str();
}

} // namespace

void runSelfReferentialExperiment(
    const DatasetSampleConfig& cfg,
    PositionAlgorithm algorithm,
    int map_frame_stride)
{
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "SAME-DOMAIN SELF-REFERENTIAL EXPERIMENT: " << cfg.sample_name << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    if (algorithm != PositionAlgorithm::ORB) {
        std::cerr << "This experiment is only validated for ORB so far -- "
                     "see CLAUDE.md Investigation Log. Aborting." << std::endl;
        return;
    }

    mkdir("CSV Files", 0755);

    // ------------------------------------------------------------------
    // STEP 1: Ensure ground truth exists, load the dense per-frame path.
    // ------------------------------------------------------------------
    TelemetryImportResult telemetry = importUavVisLocTelemetry(cfg.telemetry_csv, cfg.sample_name);
    if (!telemetry.success) {
        std::cerr << "Failed to import telemetry for " << cfg.sample_name << std::endl;
        return;
    }
    std::vector<std::pair<double, double>> gt_path = loadGroundTruthPath(cfg.sample_name);
    if (gt_path.empty()) {
        std::cerr << "No ground truth available for " << cfg.sample_name << std::endl;
        return;
    }

    // ------------------------------------------------------------------
    // STEP 2: Split into disjoint map / query frame sets by index, keeping
    // only frames with real ground truth (gt_path's {0,0} sentinel means no
    // row for that frame -- see GroundTruthAnnotator::loadGroundTruthPath).
    // ------------------------------------------------------------------
    std::vector<int> map_frames, query_frames;
    double centroid_lat_sum = 0.0, centroid_lng_sum = 0.0;
    int centroid_n = 0;

    for (int idx = 0; idx < static_cast<int>(gt_path.size()) && idx < cfg.frame_count; ++idx) {
        if (gt_path[idx].first == 0.0 && gt_path[idx].second == 0.0) continue;
        centroid_lat_sum += gt_path[idx].first;
        centroid_lng_sum += gt_path[idx].second;
        ++centroid_n;
        if (idx % map_frame_stride == 0)
            map_frames.push_back(idx);
        else
            query_frames.push_back(idx);
    }

    if (map_frames.empty() || query_frames.empty() || centroid_n == 0) {
        std::cerr << "Not enough ground-truth-tagged frames to build map/query split." << std::endl;
        return;
    }

    std::pair<double, double> fallback_position = {
        centroid_lat_sum / centroid_n, centroid_lng_sum / centroid_n};

    std::cout << "   Map frames: " << map_frames.size()
              << "  |  Query frames: " << query_frames.size()
              << "  (stride=" << map_frame_stride << ")" << std::endl;

    // ------------------------------------------------------------------
    // STEP 3: Build the reference database directly from map frames' own
    // images + real ground-truth coordinates -- no grid-cutting needed,
    // unlike generateReferenceCropsGrid() against a stitched satellite map.
    // Same preprocessing (CLAHE, no mask -- these are clean aerial survey
    // photos, see DatasetSamples.hpp) as query frames get below, so both
    // sides are treated identically.
    // ------------------------------------------------------------------
    std::vector<ReferenceCrop> reference_crops;
    reference_crops.reserve(map_frames.size());
    for (int idx : map_frames) {
        cv::Mat img = cv::imread(frameImagePath(cfg, idx + 1));
        if (img.empty()) continue;
        cv::Mat preprocessed = EdgeProcessor::preprocessFrame(img);
        reference_crops.push_back(ReferenceCrop(gt_path[idx], preprocessed));
    }
    std::cout << "   ✓ Built reference database: " << reference_crops.size() << " crops" << std::endl;

    ORBFeatureEstimator estimator;
    estimator.precomputeAll(reference_crops);

    // ------------------------------------------------------------------
    // STEP 4: Query each held-out frame independently (no Kalman/particle
    // filter fusion -- this measures single-frame match quality against a
    // same-domain reference, not fusion, which is already tested
    // separately -- see CLAUDE.md Investigation Log).
    // ------------------------------------------------------------------
    std::string csv_path = "CSV Files/video_telemetry_orb_selfref_" + cfg.sample_name + ".csv";
    std::ofstream csv(csv_path);
    csv << "Frame,Time_Sec,Raw_Lat,Raw_Lng,Predicted_Lat,Predicted_Lng,"
           "Filtered_Lat,Filtered_Lng,Match_Confidence,Innovation_M,"
           "Outlier_Rejected,Best_Match_Idx,Algorithm\n";

    int processed = 0;
    for (int idx : query_frames) {
        cv::Mat img = cv::imread(frameImagePath(cfg, idx + 1));
        if (img.empty()) continue;
        cv::Mat preprocessed = EdgeProcessor::preprocessFrame(img);

        PositionEstimate estimate = estimator.estimatePosition(
            preprocessed, reference_crops, fallback_position);

        // No fusion step in this experiment -- Filtered duplicates Raw so
        // the unmodified evaluate_ground_truth.py (which only reads
        // Raw_Lat/Lng and Filtered_Lat/Lng) works without any changes.
        csv << (idx + 1) << ","
            << std::fixed << std::setprecision(2) << 0.0 << ","
            << std::setprecision(8)
            << estimate.position.first  << "," << estimate.position.second << ","
            << estimate.position.first  << "," << estimate.position.second << ","
            << estimate.position.first  << "," << estimate.position.second << ","
            << std::setprecision(4) << estimate.confidence << ","
            << std::setprecision(2) << 0.0 << ","
            << "0,"
            << estimate.best_match_idx << ","
            << "orb\n";
        csv.flush();

        ++processed;
        if (processed % 50 == 0)
            std::cout << "   Query frame " << processed << "/" << query_frames.size() << std::endl;
    }

    std::cout << "\n✅ COMPLETE | " << processed << " query frames processed" << std::endl;
    std::cout << "   Telemetry: " << csv_path << std::endl;
    std::cout << "   Evaluate with: python3 scripts/evaluate_ground_truth.py --gt \""
              << "CSV Files/ground_truth_" << cfg.sample_name << ".csv\" --telemetry \""
              << csv_path << "\"" << std::endl;
}
