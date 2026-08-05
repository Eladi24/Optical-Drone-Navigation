#include "ORBFeatureEstimator.hpp"
#include "SyncBarrier.hpp"
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <iostream>

// =============================================================================
// Internal task data — kept in the .cpp as an implementation detail.
// Anonymous namespace prevents name collisions with identical structs that
// might be defined in other translation units (e.g. GlobalLocator.cpp).
// =============================================================================
namespace {

struct Candidate {
    int                      crop_idx;
    int                      good_match_count;
    std::vector<cv::Point2f> src_pts;   // drone-view keypoint positions
    std::vector<cv::Point2f> dst_pts;   // reference-crop keypoint positions
};

// Data block passed to each matchWorker thread via void*.
// Each thread has its own independent instance — no fields are shared between
// threads, so no additional mutex is needed on the output vector.
struct MatchTaskData {
    // Inputs (read-only, set before enqueueing)
    int                               start_idx;
    int                               end_idx;
    const std::vector<ReferenceCrop>* crops;
    const ORBFeatureEstimator*        estimator;   // for getCachedFeatures
    const cv::Mat*                    desc_drone;
    const std::vector<cv::KeyPoint>*  kp_drone;
    SyncBarrier*                      barrier;

    // Output (written exclusively by this thread)
    std::vector<Candidate>            candidates;
};

} // anonymous namespace

// =============================================================================
// Constructor / Destructor
// =============================================================================

ORBFeatureEstimator::ORBFeatureEstimator(int num_features, int num_threads)
    : orb_detector_(cv::ORB::create(num_features))
    , pool_(static_cast<size_t>(std::max(1, num_threads)))
{
    pthread_rwlock_init(&cache_rwlock_, nullptr);
}

ORBFeatureEstimator::~ORBFeatureEstimator()
{
    pthread_rwlock_destroy(&cache_rwlock_);
}

// =============================================================================
// Cache management
// =============================================================================

void ORBFeatureEstimator::clearCache()
{
    pthread_rwlock_wrlock(&cache_rwlock_);
    feature_cache_.clear();
    pthread_rwlock_unlock(&cache_rwlock_);
}

// -----------------------------------------------------------------------------
// getCachedFeatures
//
// Thread-safety contract:
//   Read path  — shared (rdlock): multiple threads may call simultaneously
//                after precomputeAll() with zero contention.
//   Write path — exclusive (wrlock): only occurs on a cache miss.
//                After precomputeAll() populates all entries, the write path
//                is never reached during the video loop.
//
// Pointer stability:
//   We call feature_cache_.reserve(N) in precomputeAll() before any insertions.
//   This guarantees the map will not rehash up to N elements, so the pointers
//   returned here remain valid for the lifetime of the estimator.
// -----------------------------------------------------------------------------
const ORBFeatureEstimator::CachedFeatures*
ORBFeatureEstimator::getCachedFeatures(const cv::Mat& image) const
{
    const uchar* key = image.data;

    // --- Fast path: shared read lock (concurrent reads, zero contention) ---
    pthread_rwlock_rdlock(&cache_rwlock_);
    auto it = feature_cache_.find(key);
    if (it != feature_cache_.end()) {
        const CachedFeatures* result = it->second.valid ? &it->second : nullptr;
        pthread_rwlock_unlock(&cache_rwlock_);
        return result;
    }
    pthread_rwlock_unlock(&cache_rwlock_);

    // --- Slow path: compute descriptors, then take exclusive write lock ---
    CachedFeatures entry;
    {
        // Each call creates its own local detector because cv::ORB is not
        // thread-safe — sharing orb_detector_ across concurrent cache misses
        // would cause data races.
        auto det = cv::ORB::create(orb_detector_->getMaxFeatures());
        det->detectAndCompute(image, cv::noArray(), entry.keypoints, entry.descriptors);
        // MIN_VALID_KEYPOINTS=150: measured across all 572 of this project's
        // Haifa reference crops (see ORB keypoint-count histogram). The
        // distribution is strongly bimodal -- 29% of crops (near-featureless
        // water, sometimes with only the tiled Google Maps watermark text
        // for texture) have 0-40 keypoints, then a thin, sparse band up to
        // ~200, then 54% of crops (genuine terrain: piers, ships, buildings)
        // have 300-500. A first attempt at 40 was set from only 3-4 spot
        // checks and was too low -- a 64-keypoint water+watermark crop still
        // got through and dominated 20% of frames in testing. 150 sits in
        // the sparse valley between the two populations.
        static constexpr size_t MIN_VALID_KEYPOINTS = 150;
        entry.valid = !entry.descriptors.empty() &&
                      entry.keypoints.size() >= MIN_VALID_KEYPOINTS;
    }

    pthread_rwlock_wrlock(&cache_rwlock_);

    // Double-check: another thread may have computed and inserted this entry
    // while we were running detectAndCompute above.
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

void ORBFeatureEstimator::precomputeAll(const std::vector<ReferenceCrop>& crops)
{
    // Reserve enough buckets for all crops before any insertions.
    // This prevents rehashing during or after precomputation, which is what
    // makes the pointers returned by getCachedFeatures stable during threading.
    pthread_rwlock_wrlock(&cache_rwlock_);
    feature_cache_.reserve(crops.size());
    pthread_rwlock_unlock(&cache_rwlock_);

    std::cout << "⚙️  ORBFeatureEstimator: pre-computing descriptors for "
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
// Runs Lowe's ratio test against a slice of the crop database and records
// every crop that produces >= 8 ratio-passing matches.  The results are
// written to the thread's own MatchTaskData::candidates vector.
//
// Declared as a static member of ORBFeatureEstimator so that it has access
// to the private CachedFeatures type and the private getCachedFeatures method
// via the estimator pointer in MatchTaskData.
// =============================================================================
void* ORBFeatureEstimator::matchWorker(void* arg)
{
    // Reinterpret the anonymous-namespace type inside a class member function.
    // This is valid: matchWorker is a member of ORBFeatureEstimator, so it can
    // see all private types.  The anonymous-namespace Candidate / MatchTaskData
    // are visible because this translation unit defines them above.
    MatchTaskData* d = static_cast<MatchTaskData*>(arg);

    // Each thread needs its own BFMatcher — cv::BFMatcher is not thread-safe.
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    for (int i = d->start_idx; i < d->end_idx; ++i) {
        const auto& crop = (*d->crops)[i];
        if (crop.image.empty()) continue;

        const CachedFeatures* ref = d->estimator->getCachedFeatures(crop.image);
        if (!ref) continue;   // featureless or invalid crop

        std::vector<std::vector<cv::DMatch>> knn;
        try {
            matcher.knnMatch(*d->desc_drone, ref->descriptors, knn, 2);
        } catch (...) { continue; }

        // Lowe's ratio test — 0.75 is the standard threshold for ORB
        std::vector<cv::DMatch> good;
        good.reserve(knn.size());
        for (const auto& m : knn)
            if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance)
                good.push_back(m[0]);

        // Homography requires at least 8 well-distributed correspondences to
        // produce a meaningful result — discard weaker matches early
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

PositionEstimate ORBFeatureEstimator::estimatePosition(
    const cv::Mat&                    drone_view,
    const std::vector<ReferenceCrop>& reference_crops,
    const std::pair<double, double>&  last_position)
{
    if (drone_view.empty() || reference_crops.empty())
        return PositionEstimate(last_position, 0.0, -1);

    // Extract features from the current drone frame.
    // Not cached — the drone view changes on every call.
    // feature_mask_ excludes logo/subtitle regions set by VideoPreprocessor.
    std::vector<cv::KeyPoint> kp_drone;
    cv::Mat                   desc_drone;
    cv::InputArray mask_input = feature_mask_.empty() ? cv::noArray() : cv::InputArray(feature_mask_);
    orb_detector_->detectAndCompute(drone_view, mask_input, kp_drone, desc_drone);

    if (kp_drone.size() < 10 || desc_drone.empty())
        return PositionEstimate(last_position, 0.3, -1);

    // =========================================================================
    // STAGE 1 — Parallel Lowe's ratio test across all candidate crops
    //
    // The crop list is split into equal-sized slices, one per pool thread.
    // Each worker writes its results into its own MatchTaskData::candidates —
    // no cross-thread sharing, no additional locking on the output side.
    //
    // The SyncBarrier blocks this thread until every worker has signalled,
    // at which point it is safe to read all per-thread results.
    // =========================================================================
    const int n_crops   = static_cast<int>(reference_crops.size());
    const int n_threads = static_cast<int>(pool_.threadCount()); // always >= 1
    const int chunk     = n_crops / n_threads;

    SyncBarrier barrier(n_threads);

    // Allocate task structs on the stack. They must outlive barrier.wait(),
    // which is guaranteed since both live in the same scope.
    std::vector<MatchTaskData> tasks(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        tasks[t].start_idx  = t * chunk;
        tasks[t].end_idx    = (t == n_threads - 1) ? n_crops : (t + 1) * chunk;
        tasks[t].crops      = &reference_crops;
        tasks[t].estimator  = this;
        tasks[t].desc_drone = &desc_drone;
        tasks[t].kp_drone   = &kp_drone;
        tasks[t].barrier    = &barrier;
        // tasks[t].candidates is default-constructed (empty vector)
        pool_.enqueue(matchWorker, &tasks[t]);
    }

    barrier.wait();  // blocks until all n_threads workers have signalled

    // Merge per-thread candidate lists into a single flat list
    std::vector<Candidate> candidates;
    for (auto& t : tasks)
        for (auto& c : t.candidates)
            candidates.push_back(std::move(c));

    if (candidates.empty())
        return PositionEstimate(last_position, 0.3, -1);

    // Sort descending by raw good-match count so RANSAC evaluates the most
    // promising crops first
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.good_match_count > b.good_match_count;
              });

    // Limit RANSAC to the top N candidates to keep per-frame latency bounded.
    // The true match almost always ranks first or second by raw count.
    const int MAX_RANSAC = 5;
    if (static_cast<int>(candidates.size()) > MAX_RANSAC)
        candidates.resize(MAX_RANSAC);

    // =========================================================================
    // STAGE 2 — RANSAC homography verification on the shortlist
    //
    // Inlier count is a stronger score than raw match count: it requires the
    // matches to be geometrically consistent as a planar transformation.
    // False positives from texture similarity that survive the ratio test
    // will not produce a coherent homography.
    //
    // Heading-consistency scoring (Phase A1) was tried here and reverted --
    // measured on flight 01, filtered mean error went from 2269m to 2407m
    // (~6% worse), raw roughly flat. Diagnosis: the homography's implied
    // rotation is only well-constrained when a candidate has enough inlier
    // points (the one calibration match that validated cleanly against Phi1
    // had 20 inliers), but most candidates never get close to that (see
    // MIN_VALID_KEYPOINTS commentary elsewhere -- strong matches are rare).
    // For weak, barely-passing homographies the implied rotation is close to
    // noise, so scoring by heading-consistency there adds noise, not signal.
    // See CLAUDE.md Investigation Log for the full writeup.
    // =========================================================================
    int best_idx     = -1;
    int best_inliers = 0;
    // Every candidate that produced a geometrically consistent homography,
    // not just the winner -- feeds ParticleFilter's multi-hypothesis
    // update() (see PositionEstimation.hpp). Harmless extra bookkeeping for
    // the Kalman path, which never reads this field.
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

    // If RANSAC found no geometrically consistent match, fall back to the
    // highest raw-count candidate with a floor confidence
    if (best_idx == -1) {
        best_idx     = candidates[0].crop_idx;
        best_inliers = 0;
    }

    // =========================================================================
    // Confidence mapping:
    //   0  inliers → 0.30  (floor: RANSAC rejected everything)
    //  15  inliers → 0.65
    //  30+ inliers → 1.00  (strong geometric lock)
    //
    // The divisor of 30 is empirical. If your scene consistently yields 8–15
    // inliers for known-good matches, lower it to ~15 to spread the range.
    // If you regularly see 40+ inliers, raise it accordingly.
    // =========================================================================
    const double confidence = (best_inliers == 0)
        ? 0.3
        : std::min(1.0, 0.3 + (best_inliers / 30.0) * 0.7);

    PositionEstimate result(reference_crops[best_idx].coordinates, confidence, best_idx);
    result.candidates = std::move(surviving_candidates);
    return result;
}