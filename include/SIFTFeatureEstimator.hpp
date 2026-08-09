#pragma once
#include "PositionEstimation.hpp"
#include "ThreadPool.hpp"
#include "SyncBarrier.hpp"
#include <opencv2/features2d.hpp>
#include <unordered_map>
#include <pthread.h>

// ---------------------------------------------------------------------------
// SIFTFeatureEstimator — parallel SIFT-based position estimator.
//
// Architecture mirrors ORBFeatureEstimator:
//   • precompute(): fills the descriptor cache in parallel before the loop.
//   • estimatePosition(): Stage 1 is a parallel Lowe's ratio test across all
//     crops; Stage 2 is sequential RANSAC homography on the short-list.
//   • getCachedFeatures(): thread-safe via pthread_rwlock_t.
//
// SIFT vs ORB tradeoffs for this application:
//   • SIFT is slower per descriptor but more distinctive — especially useful
//     on the harbour scene where ORB features cluster on wave glints.
//   • SIFT uses float (L2) descriptors; BFMatcher uses NORM_L2.
//   • SIFT is rotation- and scale-invariant, so it handles slight drone roll
//     or altitude variation more gracefully than ORB.
// ---------------------------------------------------------------------------
class SIFTFeatureEstimator : public IPositionEstimator {
public:
    // num_features: cap at 500 — unlimited SIFT is ~3× slower with marginal gain
    // num_threads:  pool workers for Stage 1 matching; match your CPU core count
    explicit SIFTFeatureEstimator(int num_features = 500, int num_threads = 4);
    ~SIFTFeatureEstimator() override;

    // Pre-compute and cache SIFT descriptors for every crop.
    // Call once before the processing loop.  After this all getCachedFeatures
    // calls during threading are guaranteed cache hits — zero contention.
    void precompute(const std::vector<ReferenceCrop>& crops) override;

    // Discard the descriptor cache (e.g. when the crop database is rebuilt).
    void clearCache();

    void setFeatureMask(const cv::Mat& mask) override { feature_mask_ = mask; }

    void setGeoReference(double mpp, double meters_per_degree_lat,
                          double meters_per_degree_lng) override {
        mpp_ = mpp;
        meters_per_degree_lat_ = meters_per_degree_lat;
        meters_per_degree_lng_ = meters_per_degree_lng;
    }

    PositionEstimate estimatePosition(
        const cv::Mat&                    drone_view,
        const std::vector<ReferenceCrop>& reference_crops,
        const std::pair<double, double>&  last_position = {0.0, 0.0}
    ) override;

    std::string getName() const override { return "sift"; }

private:
    cv::Ptr<cv::SIFT> sift_detector_;
    cv::Mat           feature_mask_;  // 255=use, 0=exclude; applied only to drone frames

    // Geo-referencing scale set via setGeoReference(); 0.0 until then, which
    // also serves as the guard that skips homography-based position
    // refinement if it was never called (see estimatePosition()).
    double mpp_ = 0.0;
    double meters_per_degree_lat_ = 0.0;
    double meters_per_degree_lng_ = 0.0;

    // Persistent thread pool — one per estimator instance, reused every frame.
    ThreadPool pool_;

    // -------------------------------------------------------------------------
    // Descriptor cache (same design as ORBFeatureEstimator).
    // Key = raw data pointer; stable because crops are .clone()d at startup.
    // pthread_rwlock_t: many concurrent reads during the video loop (no writes
    // after precompute()), with exclusive writes only on cache misses.
    // -------------------------------------------------------------------------
    struct CachedFeatures {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat                   descriptors;
        bool                      valid = false;
    };

    mutable std::unordered_map<const uchar*, CachedFeatures> feature_cache_;
    mutable pthread_rwlock_t                                   cache_rwlock_;

    // Retrieve (or lazily compute and store) features.  Thread-safe.
    const CachedFeatures* getCachedFeatures(const cv::Mat& image) const;

    // Static pthread entry point for Stage 1 (ratio test) workers.
    static void* matchWorker(void* arg);
};
