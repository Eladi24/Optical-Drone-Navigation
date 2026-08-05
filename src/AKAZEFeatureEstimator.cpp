#include "AKAZEFeatureEstimator.hpp"
#include "SyncBarrier.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <algorithm>
#include <iostream>

// =============================================================================
// Internal task data — see ORBFeatureEstimator.cpp for the full rationale,
// identical here (same threading architecture, only the detector differs).
// =============================================================================
namespace {

struct Candidate {
    int                      crop_idx;
    int                      good_match_count;
    std::vector<cv::Point2f> src_pts;
    std::vector<cv::Point2f> dst_pts;
};

struct MatchTaskData {
    int                                start_idx;
    int                                end_idx;
    const std::vector<ReferenceCrop>*  crops;
    const AKAZEFeatureEstimator*       estimator;
    const cv::Mat*                     desc_drone;
    const std::vector<cv::KeyPoint>*   kp_drone;
    SyncBarrier*                       barrier;

    std::vector<Candidate>             candidates;
};

} // anonymous namespace

// =============================================================================
// Constructor / Destructor
// =============================================================================

AKAZEFeatureEstimator::AKAZEFeatureEstimator(float detection_threshold, int num_threads)
    : detection_threshold_(detection_threshold)
    , pool_(static_cast<size_t>(std::max(1, num_threads)))
{
    pthread_rwlock_init(&cache_rwlock_, nullptr);
}

AKAZEFeatureEstimator::~AKAZEFeatureEstimator()
{
    pthread_rwlock_destroy(&cache_rwlock_);
}

// =============================================================================
// Cache management
// =============================================================================

void AKAZEFeatureEstimator::clearCache()
{
    pthread_rwlock_wrlock(&cache_rwlock_);
    feature_cache_.clear();
    pthread_rwlock_unlock(&cache_rwlock_);
}

const AKAZEFeatureEstimator::CachedFeatures*
AKAZEFeatureEstimator::getCachedFeatures(const cv::Mat& image) const
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

    CachedFeatures entry;
    {
        // Each call creates its own local detector -- cv::AKAZE is not
        // thread-safe, same reason ORBFeatureEstimator does this.
        auto det = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, detection_threshold_);
        det->detectAndCompute(image, cv::noArray(), entry.keypoints, entry.descriptors);
        // MIN_VALID_KEYPOINTS=150: inherited from ORBFeatureEstimator's
        // Haifa-crop-histogram-derived threshold (see that file), not
        // independently re-tuned for AKAZE's keypoint distribution -- AKAZE
        // is threshold-, not count-, based, so its keypoint counts per crop
        // may skew differently. Kept identical for a first, fair comparison
        // against the ORB baseline; revisit if AKAZE crops are being
        // systematically starved or over-admitted by this cutoff.
        static constexpr size_t MIN_VALID_KEYPOINTS = 150;
        entry.valid = !entry.descriptors.empty() &&
                      entry.keypoints.size() >= MIN_VALID_KEYPOINTS;
    }

    pthread_rwlock_wrlock(&cache_rwlock_);

    it = feature_cache_.find(key);
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
// precomputeAll
// =============================================================================

void AKAZEFeatureEstimator::precomputeAll(const std::vector<ReferenceCrop>& crops)
{
    pthread_rwlock_wrlock(&cache_rwlock_);
    feature_cache_.reserve(crops.size());
    pthread_rwlock_unlock(&cache_rwlock_);

    std::cout << "⚙️  AKAZEFeatureEstimator: pre-computing descriptors for "
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
// matchWorker — Stage 1 pthread entry point (identical structure to ORB's,
// NORM_HAMMING is correct here too since MLDB is a binary descriptor)
// =============================================================================
void* AKAZEFeatureEstimator::matchWorker(void* arg)
{
    MatchTaskData* d = static_cast<MatchTaskData*>(arg);

    cv::BFMatcher matcher(cv::NORM_HAMMING);

    for (int i = d->start_idx; i < d->end_idx; ++i) {
        const auto& crop = (*d->crops)[i];
        if (crop.image.empty()) continue;

        const CachedFeatures* ref = d->estimator->getCachedFeatures(crop.image);
        if (!ref) continue;

        std::vector<std::vector<cv::DMatch>> knn;
        try {
            matcher.knnMatch(*d->desc_drone, ref->descriptors, knn, 2);
        } catch (...) { continue; }

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
// estimatePosition — identical structure/scoring to ORBFeatureEstimator
// =============================================================================

PositionEstimate AKAZEFeatureEstimator::estimatePosition(
    const cv::Mat&                    drone_view,
    const std::vector<ReferenceCrop>& reference_crops,
    const std::pair<double, double>&  last_position)
{
    if (drone_view.empty() || reference_crops.empty())
        return PositionEstimate(last_position, 0.0, -1);

    std::vector<cv::KeyPoint> kp_drone;
    cv::Mat                   desc_drone;
    cv::InputArray mask_input = feature_mask_.empty() ? cv::noArray() : cv::InputArray(feature_mask_);
    auto det = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, detection_threshold_);
    det->detectAndCompute(drone_view, mask_input, kp_drone, desc_drone);

    if (kp_drone.size() < 10 || desc_drone.empty())
        return PositionEstimate(last_position, 0.3, -1);

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

    std::vector<Candidate> candidates;
    for (auto& t : tasks)
        for (auto& c : t.candidates)
            candidates.push_back(std::move(c));

    if (candidates.empty())
        return PositionEstimate(last_position, 0.3, -1);

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.good_match_count > b.good_match_count;
              });

    const int MAX_RANSAC = 5;
    if (static_cast<int>(candidates.size()) > MAX_RANSAC)
        candidates.resize(MAX_RANSAC);

    int best_idx     = -1;
    int best_inliers = 0;
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
        }
    }

    if (best_idx == -1) {
        best_idx     = candidates[0].crop_idx;
        best_inliers = 0;
    }

    const double confidence = (best_inliers == 0)
        ? 0.3
        : std::min(1.0, 0.3 + (best_inliers / 30.0) * 0.7);

    PositionEstimate result(reference_crops[best_idx].coordinates, confidence, best_idx);
    result.candidates = std::move(surviving_candidates);
    return result;
}
