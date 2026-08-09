#include "SIFTFeatureEstimator.hpp"
#include "CoordinateUtils.hpp"
#include "VideoProcessing.hpp"  // kReferenceCropMeters
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <iostream>

// =============================================================================
// Internal task data — anonymous namespace prevents collisions with identical
// structs in other translation units (GlobalLocator, ORBFeatureEstimator).
// =============================================================================
namespace {

struct Candidate {
    int                      crop_idx;
    int                      good_match_count;
    std::vector<cv::Point2f> src_pts;
    std::vector<cv::Point2f> dst_pts;
};

struct MatchTaskData {
    int                               start_idx;
    int                               end_idx;
    const std::vector<ReferenceCrop>* crops;
    const SIFTFeatureEstimator*       estimator;
    const cv::Mat*                    desc_drone;
    const std::vector<cv::KeyPoint>*  kp_drone;
    SyncBarrier*                      barrier;
    std::vector<Candidate>            candidates;   // written exclusively by this thread
};

} // anonymous namespace

// =============================================================================
// Constructor / Destructor
// =============================================================================

SIFTFeatureEstimator::SIFTFeatureEstimator(int num_features, int num_threads)
    : sift_detector_(cv::SIFT::create(num_features))
    , pool_(static_cast<size_t>(std::max(1, num_threads)))
{
    pthread_rwlock_init(&cache_rwlock_, nullptr);
}

SIFTFeatureEstimator::~SIFTFeatureEstimator()
{
    pthread_rwlock_destroy(&cache_rwlock_);
}

// =============================================================================
// clearCache
// =============================================================================

void SIFTFeatureEstimator::clearCache()
{
    pthread_rwlock_wrlock(&cache_rwlock_);
    feature_cache_.clear();
    pthread_rwlock_unlock(&cache_rwlock_);
}

// =============================================================================
// getCachedFeatures
//
// Thread-safety contract mirrors ORBFeatureEstimator:
//   Read path  — shared rdlock (zero contention after precompute()).
//   Write path — exclusive wrlock (only on a cache miss).
//
// SIFT is not thread-safe, so each cache-miss call creates a local detector.
// =============================================================================
const SIFTFeatureEstimator::CachedFeatures*
SIFTFeatureEstimator::getCachedFeatures(const cv::Mat& image) const
{
    const uchar* key = image.data;

    pthread_rwlock_rdlock(&cache_rwlock_);
    auto it = feature_cache_.find(key);
    if (it != feature_cache_.end()) {
        const CachedFeatures* result = it->second.valid ? &it->second : nullptr;
        pthread_rwlock_unlock(&cache_rwlock_);
        return result;
    }
    pthread_rwlock_unlock(&cache_rwlock_);

    // Cache miss — compute descriptors with a thread-local detector
    CachedFeatures entry;
    {
        auto det = cv::SIFT::create(sift_detector_->getNFeatures());
        det->detectAndCompute(image, cv::noArray(), entry.keypoints, entry.descriptors);
        // MIN_VALID_KEYPOINTS=150: see ORBFeatureEstimator.cpp -- measured
        // across all 572 of this project's Haifa reference crops with SIFT.
        // 207 crops (near-featureless water, sometimes with only the tiled
        // Google Maps watermark text for texture) fall below 100 keypoints;
        // 365 crops (genuine terrain) sit above 150, many saturating the
        // 500-feature cap entirely. 150 sits in the gap between the two.
        static constexpr size_t MIN_VALID_KEYPOINTS = 150;
        entry.valid = !entry.descriptors.empty() &&
                      entry.keypoints.size() >= MIN_VALID_KEYPOINTS;
    }

    pthread_rwlock_wrlock(&cache_rwlock_);
    it = feature_cache_.find(key);   // double-check after lock upgrade
    if (it != feature_cache_.end()) {
        const CachedFeatures* result = it->second.valid ? &it->second : nullptr;
        pthread_rwlock_unlock(&cache_rwlock_);
        return result;
    }
    auto [ins, _] = feature_cache_.emplace(key, std::move(entry));
    const CachedFeatures* result = ins->second.valid ? &ins->second : nullptr;
    pthread_rwlock_unlock(&cache_rwlock_);
    return result;
}

// =============================================================================
// precompute — override of IPositionEstimator::precompute()
//
// Pre-fills the descriptor cache for all crops before the video loop.
// After this call all getCachedFeatures() accesses are guaranteed read-only,
// so Stage 1 threads can run with zero lock contention.
// =============================================================================
void SIFTFeatureEstimator::precompute(const std::vector<ReferenceCrop>& crops)
{
    pthread_rwlock_wrlock(&cache_rwlock_);
    feature_cache_.reserve(crops.size());
    pthread_rwlock_unlock(&cache_rwlock_);

    std::cout << "⚙️  SIFTFeatureEstimator: pre-computing descriptors for "
              << crops.size() << " crops... ";
    std::cout.flush();

    int valid = 0;
    for (const auto& crop : crops) {
        if (!crop.image.empty()) {
            const CachedFeatures* f = getCachedFeatures(crop.image);
            if (f) ++valid;
        }
    }

    std::cout << "done (" << valid << " valid)." << std::endl;
}

// =============================================================================
// matchWorker — Stage 1 pthread entry point
//
// Runs Lowe's ratio test (threshold 0.75) against a slice of the crop
// database.  Results written to the thread's own MatchTaskData::candidates.
// =============================================================================
void* SIFTFeatureEstimator::matchWorker(void* arg)
{
    MatchTaskData* d = static_cast<MatchTaskData*>(arg);

    cv::BFMatcher matcher(cv::NORM_L2);   // SIFT descriptors → Euclidean distance

    for (int i = d->start_idx; i < d->end_idx; ++i) {
        const auto& crop = (*d->crops)[i];
        if (crop.image.empty()) continue;

        const CachedFeatures* ref = d->estimator->getCachedFeatures(crop.image);
        if (!ref) continue;

        std::vector<std::vector<cv::DMatch>> knn;
        try {
            matcher.knnMatch(*d->desc_drone, ref->descriptors, knn, 2);
        } catch (...) { continue; }

        // Lowe's ratio test — 0.75 is standard for SIFT
        std::vector<cv::DMatch> good;
        good.reserve(knn.size());
        for (const auto& m : knn)
            if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance)
                good.push_back(m[0]);

        if (static_cast<int>(good.size()) < 8) continue;

        Candidate c;
        c.crop_idx         = i;
        c.good_match_count = static_cast<int>(good.size());
        c.src_pts.reserve(good.size());
        c.dst_pts.reserve(good.size());
        for (const auto& m : good) {
            c.src_pts.push_back((*d->kp_drone)[m.queryIdx].pt);
            c.dst_pts.push_back(ref->keypoints[m.trainIdx].pt);
        }
        d->candidates.push_back(std::move(c));
    }

    d->barrier->signal();
    return nullptr;
}

// =============================================================================
// estimatePosition
// =============================================================================

PositionEstimate SIFTFeatureEstimator::estimatePosition(
    const cv::Mat&                    drone_view,
    const std::vector<ReferenceCrop>& reference_crops,
    const std::pair<double, double>&  last_position)
{
    if (drone_view.empty() || reference_crops.empty())
        return PositionEstimate(last_position, 0.0, -1);

    // Extract features from the current drone frame (not cached — changes every call).
    // feature_mask_ excludes logo/subtitle regions set by VideoPreprocessor.
    std::vector<cv::KeyPoint> kp_drone;
    cv::Mat                   desc_drone;
    cv::InputArray mask_input = feature_mask_.empty() ? cv::noArray() : cv::InputArray(feature_mask_);
    sift_detector_->detectAndCompute(drone_view, mask_input, kp_drone, desc_drone);

    // Fewer than 15 keypoints → featureless water/haze; return low confidence
    if (kp_drone.size() < 15 || desc_drone.empty())
        return PositionEstimate(last_position, 0.1, -1);

    // =========================================================================
    // STAGE 1 — Parallel Lowe's ratio test across all candidate crops
    // =========================================================================
    const int n_crops   = static_cast<int>(reference_crops.size());
    const int n_threads = static_cast<int>(pool_.threadCount());
    const int chunk     = n_crops / n_threads;

    SyncBarrier barrier(n_threads);
    std::vector<MatchTaskData> tasks(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        tasks[t].start_idx  = t * chunk;
        tasks[t].end_idx    = (t == n_threads - 1) ? n_crops : (t + 1) * chunk;
        tasks[t].crops      = &reference_crops;
        tasks[t].estimator  = this;
        tasks[t].desc_drone = &desc_drone;
        tasks[t].kp_drone   = &kp_drone;
        tasks[t].barrier    = &barrier;
        pool_.enqueue(matchWorker, &tasks[t]);
    }

    barrier.wait();

    // Merge per-thread candidate lists
    std::vector<Candidate> candidates;
    for (auto& t : tasks)
        for (auto& c : t.candidates)
            candidates.push_back(std::move(c));

    if (candidates.empty())
        return PositionEstimate(last_position, 0.1, -1);

    // =========================================================================
    // STAGE 2 — RANSAC homography verification on every Stage-1 candidate
    //
    // Previously ran only on the top 5 by raw match count. Raw count is
    // exactly the statistic cross-domain aliasing corrupts, so a
    // geometrically correct crop ranked 6th+ never reached RANSAC. Stage 1
    // already gates candidates at >=8 ratio-passing matches, so every
    // candidate that cleared it gets RANSAC-verified here. See
    // ORBFeatureEstimator.cpp for the full writeup (identical fix, mirrored).
    // =========================================================================
    int best_idx     = -1;
    int best_inliers = 0;
    cv::Mat best_H;
    // Every candidate that produced a geometrically consistent homography,
    // not just the winner -- feeds ParticleFilter's multi-hypothesis
    // update() (see PositionEstimation.hpp).
    std::vector<std::pair<std::pair<double, double>, double>> surviving_candidates;

    for (const auto& c : candidates) {
        cv::Mat mask;
        cv::Mat H = cv::findHomography(c.src_pts, c.dst_pts, cv::RANSAC, 5.0, mask);
        if (H.empty()) continue;

        int inliers = cv::countNonZero(mask);
        if (inliers == 0) continue;

        surviving_candidates.push_back({reference_crops[c.crop_idx].coordinates,
                                         static_cast<double>(inliers)});

        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_idx     = c.crop_idx;
            best_H       = H.clone();
        }
    }

    // No candidate produced a geometrically consistent homography -- no real
    // measurement this frame. Report it honestly instead of falling back to
    // an unverified raw-count guess; let the Kalman filter predict.
    if (best_idx == -1) {
        PositionEstimate result(last_position, 0.0, -1);
        result.measurement_valid = false;
        return result;
    }

    // Continuous position refinement — see ORBFeatureEstimator.cpp for the
    // full rationale (identical fix, mirrored here).
    std::pair<double, double> refined_position = reference_crops[best_idx].coordinates;
    if (mpp_ > 0.0) {
        const cv::Mat& crop_image = reference_crops[best_idx].image;
        std::vector<cv::Point2f> src{cv::Point2f(drone_view.cols / 2.0f, drone_view.rows / 2.0f)};
        std::vector<cv::Point2f> dst;
        cv::perspectiveTransform(src, dst, best_H);

        double offset_px_x = std::abs(dst[0].x - crop_image.cols / 2.0);
        double offset_px_y = std::abs(dst[0].y - crop_image.rows / 2.0);
        double offset_m_x  = offset_px_x * mpp_;
        double offset_m_y  = offset_px_y * mpp_;

        if (offset_m_x <= kReferenceCropMeters * 0.75 && offset_m_y <= kReferenceCropMeters * 0.75) {
            refined_position = CoordinateUtils::pixelToLatLng(
                static_cast<int>(dst[0].x), static_cast<int>(dst[0].y),
                reference_crops[best_idx].coordinates.first,
                reference_crops[best_idx].coordinates.second,
                crop_image.cols / 2, crop_image.rows / 2,
                mpp_, meters_per_degree_lat_, meters_per_degree_lng_);
        }
    }

    // =========================================================================
    // Confidence mapping
    //   0  inliers → 0.10
    //  10  inliers → 0.45
    //  20  inliers → 0.80   ← reliable SIFT lock
    //  30+ inliers → 1.00
    //
    // SIFT produces more geometrically precise matches than ORB; the divisor
    // of 25 spreads the range nicely over typical inlier counts for this scene.
    // =========================================================================
    const double confidence = std::min(1.0, 0.1 + (best_inliers / 25.0) * 0.9);

    PositionEstimate result(refined_position, confidence, best_idx);
    result.candidates = std::move(surviving_candidates);
    result.inliers     = best_inliers;
    return result;
}
