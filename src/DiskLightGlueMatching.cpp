#include "DiskLightGlueMatching.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>

namespace {
// Must match scripts/export_disk_lightglue.py's INPUT_SIZE/MAX_KEYPOINTS exactly.
constexpr int kInputSize   = 384;
constexpr int kDescDim     = 128;

// Same threshold value and rationale as XFeatMatching/OrbRansacMatching/ORBFeatureEstimator's
// MIN_VALID_KEYPOINTS -- inherited unmodified, not independently re-tuned for DISK's own
// NMS-based, threshold-0.0 keypoint distribution (same caveat XFeatMatching/AKAZEFeatureEstimator
// already document for their own inherited use of this constant).
constexpr size_t kMinValidKeypoints = 150;

// CPU-only LightGlue inference (9-layer self+cross-attention over up to 1024 keypoints) is slow
// enough per candidate that a full multi-flight gate was impractical either way this was tried:
// single-threaded ONNX Runtime intra-op processed roughly 1 frame per ~150s on flight 10 (the
// smallest flight); a 4-intra-op-thread version, tried first as the obvious lever, measured
// WORSE (barely past 1 frame in 120s wall time despite burning 7m26s of CPU time getting there)
// -- thread-pool synchronization overhead across many small sequential attention-layer ops
// likely dominates over the parallelism gained, at least on this WSL2 environment; not
// independently isolated further. Left at 1 (matching XFeatMatching's own default) since more
// intra-op threads measurably hurt here. The lever that actually helps is candidate-level
// parallelism instead -- see isThreadSafe()/cache_rwlock_ below and
// SplitPipelineEstimator.cpp's threaded candidate loop, which exploits the real per-frame
// independent work (up to top_k candidates) across multiple CPU cores rather than trying to
// parallelize inside one already-small transformer forward pass.
Ort::SessionOptions makeSessionOptions()
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    return opts;
}
}  // namespace

DiskLightGlueMatching::DiskLightGlueMatching(const std::string& extractor_path,
                                              const std::string& lightglue_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "DiskLightGlueMatching")
    , extractor_session_(env_, extractor_path.c_str(), makeSessionOptions())
    , lightglue_session_(env_, lightglue_path.c_str(), makeSessionOptions())
    , mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    pthread_rwlock_init(&cache_rwlock_, nullptr);
}

DiskLightGlueMatching::~DiskLightGlueMatching()
{
    pthread_rwlock_destroy(&cache_rwlock_);
}

DiskLightGlueMatching::ExtractedFeatures DiskLightGlueMatching::extract(const cv::Mat& image,
                                                                          const cv::Mat& mask)
{
    ExtractedFeatures result;
    if (image.empty()) return result;

    cv::Mat working = image;
    if (!mask.empty() && mask.size() == image.size()) {
        working = image.clone();
        working.setTo(cv::Scalar(0, 0, 0), mask == 0);
    }

    cv::Mat resized;
    cv::resize(working, resized, cv::Size(kInputSize, kInputSize));
    // DISK expects RGB, not BGR -- confirmed against lightglue_onnx's own load_image()/
    // read_image() (cv2 read -> image[..., ::-1]), unlike XFeat, whose backbone averages
    // channels to grayscale in its very first op and so never cared about channel order.
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::Mat f32;
    rgb.convertTo(f32, CV_32F, 1.0 / 255.0);

    // HWC -> CHW.
    std::vector<float> input(static_cast<size_t>(3) * kInputSize * kInputSize);
    const int plane = kInputSize * kInputSize;
    for (int y = 0; y < kInputSize; ++y) {
        const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);
        for (int x = 0; x < kInputSize; ++x)
            for (int c = 0; c < 3; ++c)
                input[static_cast<size_t>(c) * plane + y * kInputSize + x] = row[x][c];
    }

    std::array<int64_t, 4> shape{1, 3, kInputSize, kInputSize};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_, input.data(), input.size(), shape.data(), shape.size());

    const char* input_names[]  = {"image"};
    const char* output_names[] = {"keypoints", "scores", "descriptors"};
    auto outputs = extractor_session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                           output_names, 3);

    // keypoints is int64 (DISK's own NMS selects them via nonzero(), which is naturally
    // integer-valued -- confirmed directly against the exported graph's own output dtype, not
    // assumed float like XFeat's dense-grid keypoints). descriptors is float, already
    // L2-normalized by the model itself.
    const int64_t n = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    if (n <= 0) return result;

    const int64_t* kpts_data = outputs[0].GetTensorData<int64_t>();  // [1, n, 2]
    const float* desc_data   = outputs[2].GetTensorData<float>();    // [1, n, 128]

    const float scale_x = static_cast<float>(image.cols) / kInputSize;
    const float scale_y = static_cast<float>(image.rows) / kInputSize;
    const float norm_shift_x = kInputSize / 2.0f;
    const float norm_shift_y = kInputSize / 2.0f;
    const float norm_scale   = kInputSize / 2.0f;  // max(kInputSize, kInputSize)/2, square input

    result.keypoints_orig.reserve(n);
    result.keypoints_norm.reserve(n);
    std::vector<float> desc_rows;
    desc_rows.reserve(static_cast<size_t>(n) * kDescDim);

    for (int64_t i = 0; i < n; ++i) {
        const float kx = static_cast<float>(kpts_data[i * 2]);
        const float ky = static_cast<float>(kpts_data[i * 2 + 1]);
        cv::Point2f pt_orig(kx * scale_x, ky * scale_y);

        // Hard mask filter on the rescaled ORIGINAL-space position -- same contract as
        // XFeatMatching/ORB (see their own comments): the pixel-zeroing above only discourages
        // detections there, doesn't guarantee none slip through.
        if (!mask.empty() && mask.size() == image.size()) {
            int mx = cv::borderInterpolate(static_cast<int>(pt_orig.x), mask.cols, cv::BORDER_REPLICATE);
            int my = cv::borderInterpolate(static_cast<int>(pt_orig.y), mask.rows, cv::BORDER_REPLICATE);
            if (mask.at<uchar>(my, mx) == 0) continue;
        }

        result.keypoints_orig.push_back(pt_orig);
        result.keypoints_norm.emplace_back((kx - norm_shift_x) / norm_scale,
                                            (ky - norm_shift_y) / norm_scale);
        const float* d = desc_data + static_cast<size_t>(i) * kDescDim;
        desc_rows.insert(desc_rows.end(), d, d + kDescDim);
    }

    if (!result.keypoints_orig.empty()) {
        result.descriptors = cv::Mat(static_cast<int>(result.keypoints_orig.size()), kDescDim, CV_32F,
                                      desc_rows.data()).clone();
    }
    result.valid = result.keypoints_orig.size() >= kMinValidKeypoints;
    return result;
}

const DiskLightGlueMatching::ExtractedFeatures* DiskLightGlueMatching::getCachedFeatures(
    const cv::Mat& image, const cv::Mat& mask)
{
    const uchar* key = image.data;

    // --- Fast path: shared read lock (concurrent reads, zero contention). ---
    pthread_rwlock_rdlock(&cache_rwlock_);
    auto it = feature_cache_.find(key);
    if (it != feature_cache_.end()) {
        const ExtractedFeatures* result = it->second.valid ? &it->second : nullptr;
        pthread_rwlock_unlock(&cache_rwlock_);
        return result;
    }
    pthread_rwlock_unlock(&cache_rwlock_);

    // --- Slow path: run the (unlocked) extraction, then take the exclusive write lock. ---
    ExtractedFeatures entry = extract(image, mask);

    pthread_rwlock_wrlock(&cache_rwlock_);
    // Double-check: another thread may have computed and inserted this same key while this
    // thread was running extract() above.
    it = feature_cache_.find(key);
    if (it != feature_cache_.end()) {
        const ExtractedFeatures* result = it->second.valid ? &it->second : nullptr;
        pthread_rwlock_unlock(&cache_rwlock_);
        return result;
    }
    auto [ins, _] = feature_cache_.emplace(key, std::move(entry));
    const ExtractedFeatures* result = ins->second.valid ? &ins->second : nullptr;
    pthread_rwlock_unlock(&cache_rwlock_);
    return result;
}

MatchResult DiskLightGlueMatching::match(const cv::Mat& frame, const ReferenceCrop& crop)
{
    MatchResult result;
    if (frame.empty() || crop.image.empty()) return result;

    const ExtractedFeatures* qf = getCachedFeatures(frame, feature_mask_);
    if (!qf) return result;

    const ExtractedFeatures* rf = getCachedFeatures(crop.image);  // no mask for reference crops
    if (!rf) return result;

    const int64_t n0 = static_cast<int64_t>(qf->keypoints_norm.size());
    const int64_t n1 = static_cast<int64_t>(rf->keypoints_norm.size());

    std::vector<float> kpts0(static_cast<size_t>(n0) * 2), kpts1(static_cast<size_t>(n1) * 2);
    for (int64_t i = 0; i < n0; ++i) {
        kpts0[i * 2]     = qf->keypoints_norm[i].x;
        kpts0[i * 2 + 1] = qf->keypoints_norm[i].y;
    }
    for (int64_t i = 0; i < n1; ++i) {
        kpts1[i * 2]     = rf->keypoints_norm[i].x;
        kpts1[i * 2 + 1] = rf->keypoints_norm[i].y;
    }

    std::array<int64_t, 3> kpts0_shape{1, n0, 2};
    std::array<int64_t, 3> kpts1_shape{1, n1, 2};
    std::array<int64_t, 3> desc0_shape{1, n0, kDescDim};
    std::array<int64_t, 3> desc1_shape{1, n1, kDescDim};

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_info_, kpts0.data(), kpts0.size(),
                                                       kpts0_shape.data(), kpts0_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_info_, kpts1.data(), kpts1.size(),
                                                       kpts1_shape.data(), kpts1_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_info_, const_cast<float*>(qf->descriptors.ptr<float>()),
        static_cast<size_t>(n0) * kDescDim, desc0_shape.data(), desc0_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_info_, const_cast<float*>(rf->descriptors.ptr<float>()),
        static_cast<size_t>(n1) * kDescDim, desc1_shape.data(), desc1_shape.size()));

    const char* input_names[]  = {"kpts0", "kpts1", "desc0", "desc1"};
    const char* output_names[] = {"matches0", "matches1", "mscores0", "mscores1"};

    std::vector<cv::Point2f> src_pts, dst_pts;
    try {
        auto outputs = lightglue_session_.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(),
                                               inputs.size(), output_names, 4);
        // matches0[i] is the index into image1's (crop's) keypoints, or -1 if unmatched --
        // already filtered by the model's own filter_threshold=0.1, baked into the graph at
        // export time. No separate ratio-test/cosine-threshold logic needed here, unlike
        // XFeatMatching's C++-side mutual-NN+cosine matching.
        const int64_t* matches0 = outputs[0].GetTensorData<int64_t>();
        src_pts.reserve(n0);
        dst_pts.reserve(n0);
        for (int64_t i = 0; i < n0; ++i) {
            const int64_t j = matches0[i];
            if (j < 0 || j >= n1) continue;
            src_pts.push_back(qf->keypoints_orig[i]);
            dst_pts.push_back(rf->keypoints_orig[j]);
        }
    } catch (...) {
        return result;
    }

    if (static_cast<int>(src_pts.size()) < 8) return result;

    cv::Mat mask;
    cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 5.0, mask);
    if (H.empty()) return result;

    int inliers = cv::countNonZero(mask);
    if (inliers == 0) return result;

    result.homography = H;
    result.inliers = inliers;
    return result;
}
