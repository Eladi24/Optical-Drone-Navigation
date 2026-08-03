#pragma once
#include "PositionEstimation.hpp"
#include "ThreadPool.hpp"
#include <unordered_map>
#include <pthread.h>

// Same architecture as ORBFeatureEstimator (binary MLDB descriptor, Hamming
// matching -- unlike SIFT's float/L2 descriptors) -- see that class for the
// full threading/caching rationale, which is identical here. Added
// specifically to test whether descriptor choice, not scene content or map
// resolution, explains the residual UAV-VisLoc/Haifa matching failure (see
// CLAUDE.md Investigation Log's "Cross-flight verdict" -- both other
// explanations were directly tested and ruled out as the dominant driver).
class AKAZEFeatureEstimator : public IPositionEstimator {
public:
    // detection_threshold: AKAZE's keypoint response threshold (lower = more
    // keypoints). 0.001f is OpenCV's own default. Unlike ORB, AKAZE has no
    // "max features" knob -- it's threshold-, not count-, based.
    // num_threads: workers for parallel Stage 1 matching, same role as ORB's.
    explicit AKAZEFeatureEstimator(float detection_threshold = 0.001f, int num_threads = 4);
    ~AKAZEFeatureEstimator();

    void precomputeAll(const std::vector<ReferenceCrop>& crops);
    void clearCache();

    void setFeatureMask(const cv::Mat& mask) override { feature_mask_ = mask; }

    PositionEstimate estimatePosition(
        const cv::Mat&                    drone_view,
        const std::vector<ReferenceCrop>& reference_crops,
        const std::pair<double, double>&  last_position = {0.0, 0.0}
    ) override;

    std::string getName() const override { return "akaze"; }

private:
    float            detection_threshold_;
    cv::Mat          feature_mask_;

    ThreadPool pool_;

    struct CachedFeatures {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat                   descriptors;
        bool                      valid = false;
    };

    mutable std::unordered_map<const uchar*, CachedFeatures> feature_cache_;
    mutable pthread_rwlock_t                                   cache_rwlock_;

    const CachedFeatures* getCachedFeatures(const cv::Mat& image) const;

    static void* matchWorker(void* arg);
};
