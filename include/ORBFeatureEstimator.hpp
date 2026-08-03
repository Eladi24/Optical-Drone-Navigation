#pragma once
#include "PositionEstimation.hpp"
#include "ThreadPool.hpp"
#include <unordered_map>
#include <pthread.h>

class ORBFeatureEstimator : public IPositionEstimator {
public:
    // num_features: ORB keypoints per image. 500 is fast and reliable.
    // num_threads:  workers for parallel Stage 1 matching.
    //               Should match the number of logical CPU cores available.
    explicit ORBFeatureEstimator(int num_features = 500, int num_threads = 4);
    ~ORBFeatureEstimator();

    // Pre-compute and cache ORB descriptors for every crop in the database.
    // Call this once before the video loop. After this call all getCachedFeatures
    // lookups during threading are guaranteed cache hits (read-only, no contention).
    void precomputeAll(const std::vector<ReferenceCrop>& crops);

    // Discard the descriptor cache (e.g. when the crop database is rebuilt).
    void clearCache();

    void setFeatureMask(const cv::Mat& mask) override { feature_mask_ = mask; }

    PositionEstimate estimatePosition(
        const cv::Mat&                    drone_view,
        const std::vector<ReferenceCrop>& reference_crops,
        const std::pair<double, double>&  last_position = {0.0, 0.0}
    ) override;

    std::string getName() const override { return "orb"; }

private:
    cv::Ptr<cv::ORB> orb_detector_;
    cv::Mat          feature_mask_;  // 255=use, 0=exclude; applied only to drone frames

    // Persistent thread pool — created once, reused across every video frame.
    // Avoids the pthread_create / pthread_join overhead that would otherwise
    // occur on every frame.
    ThreadPool pool_;

    // -------------------------------------------------------------------------
    // Descriptor cache
    //
    // Key: raw data pointer of the crop's cv::Mat image.
    // Safe assumption: reference crops are built with .clone() at startup and
    // their cv::Mat data buffers are never reallocated, so pointers are stable
    // for the entire run.
    //
    // pthread_rwlock_t is used instead of std::shared_mutex to stay consistent
    // with the POSIX threading model used throughout this codebase.
    // Multiple readers (matching threads) acquire the read lock concurrently;
    // a write (first-time computation of a new crop) takes an exclusive lock.
    // After precomputeAll() completes there are no more writes, so all lock
    // acquisitions during the video loop are uncontended read locks — negligible
    // overhead.
    // -------------------------------------------------------------------------
    struct CachedFeatures {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat                   descriptors;
        bool                      valid = false;
    };

    mutable std::unordered_map<const uchar*, CachedFeatures> feature_cache_;
    mutable pthread_rwlock_t                                   cache_rwlock_;

    // Returns a stable pointer into feature_cache_, or nullptr if the image
    // produces fewer than 10 keypoints.  Thread-safe via cache_rwlock_.
    const CachedFeatures* getCachedFeatures(const cv::Mat& image) const;

    // Static pthread entry point for Stage 1 matching workers.
    // Declared as a class member so the implementation can access private types.
    static void* matchWorker(void* arg);
};