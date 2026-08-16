#include "XFeatMatching.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>

namespace {
// Must match scripts/export_xfeat.py's INPUT_SIZE/TOP_K exactly.
constexpr int kInputSize = 384;
constexpr int kTopK      = 1024;
constexpr int kDescDim   = 64;

// Same threshold value and rationale as OrbRansacMatching/ORBFeatureEstimator's
// MIN_VALID_KEYPOINTS -- inherited unmodified, not independently re-tuned for
// XFeat's different (dense-grid, score-threshold-based) detection distribution,
// same caveat AKAZEFeatureEstimator already documents for its own inherited use
// of this constant.
constexpr size_t kMinValidKeypoints = 150;

Ort::SessionOptions makeSessionOptions()
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    return opts;
}
}  // namespace

XFeatMatching::XFeatMatching(const std::string& model_path, float score_threshold)
    : env_(ORT_LOGGING_LEVEL_WARNING, "XFeatMatching")
    , session_(env_, model_path.c_str(), makeSessionOptions())
    , mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , score_threshold_(score_threshold)
{
}

XFeatMatching::ExtractedFeatures XFeatMatching::extract(const cv::Mat& image, const cv::Mat& mask)
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
    cv::Mat f32;
    resized.convertTo(f32, CV_32F, 1.0 / 255.0);  // matches XFeat's own parse_input() convention

    // HWC -> CHW. Channel order (BGR vs RGB) is irrelevant here: the network's
    // very first op averages across channels to grayscale (see model.py's
    // forward(): x.mean(dim=1)), so this doesn't need a BGR->RGB conversion
    // the way OnnxDeitRetrieval's ImageNet-pretrained ViT does.
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
    const char* output_names[] = {"keypoints", "descriptors", "scores"};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                 output_names, 3);

    const float* kpts_data  = outputs[0].GetTensorData<float>();  // [1, kTopK, 2]
    const float* desc_data  = outputs[1].GetTensorData<float>();  // [1, kTopK, 64]
    const float* score_data = outputs[2].GetTensorData<float>();  // [1, kTopK]

    // Keypoint positions come back in the kInputSize x kInputSize resized
    // frame's coordinate space -- rescale to the ORIGINAL image's pixel space
    // (the crop/frame passed in, whatever size it actually is) so downstream
    // findHomography/RANSAC operates in real crop-pixel coordinates, exactly
    // like ORB's own detectAndCompute() does on the unresized image directly.
    const float scale_x = static_cast<float>(image.cols) / kInputSize;
    const float scale_y = static_cast<float>(image.rows) / kInputSize;

    result.keypoints.reserve(kTopK);
    std::vector<float> desc_rows;
    desc_rows.reserve(static_cast<size_t>(kTopK) * kDescDim);

    for (int i = 0; i < kTopK; ++i) {
        float score = score_data[i];
        if (score < score_threshold_) continue;

        cv::Point2f pt(kpts_data[i * 2] * scale_x, kpts_data[i * 2 + 1] * scale_y);

        // Hard mask filter on the rescaled ORIGINAL-space position -- matches
        // ORB's exact "mask excludes this region from detection" contract
        // (the pixel-zeroing above only discourages detections there, it
        // doesn't guarantee none slip through at a dense grid cell's edge).
        if (!mask.empty() && mask.size() == image.size()) {
            int mx = cv::borderInterpolate(static_cast<int>(pt.x), mask.cols, cv::BORDER_REPLICATE);
            int my = cv::borderInterpolate(static_cast<int>(pt.y), mask.rows, cv::BORDER_REPLICATE);
            if (mask.at<uchar>(my, mx) == 0) continue;
        }

        result.keypoints.push_back(pt);
        const float* d = desc_data + static_cast<size_t>(i) * kDescDim;
        desc_rows.insert(desc_rows.end(), d, d + kDescDim);
    }

    if (!result.keypoints.empty()) {
        result.descriptors = cv::Mat(static_cast<int>(result.keypoints.size()), kDescDim, CV_32F,
                                      desc_rows.data()).clone();
    }
    result.valid = result.keypoints.size() >= kMinValidKeypoints;
    return result;
}

const XFeatMatching::ExtractedFeatures* XFeatMatching::getCachedFeatures(const cv::Mat& image,
                                                                          const cv::Mat& mask)
{
    const uchar* key = image.data;
    auto it = feature_cache_.find(key);
    if (it != feature_cache_.end())
        return it->second.valid ? &it->second : nullptr;

    auto [ins, _] = feature_cache_.emplace(key, extract(image, mask));
    return ins->second.valid ? &ins->second : nullptr;
}

MatchResult XFeatMatching::match(const cv::Mat& frame, const ReferenceCrop& crop)
{
    MatchResult result;
    if (frame.empty() || crop.image.empty()) return result;

    const ExtractedFeatures* qf = getCachedFeatures(frame, feature_mask_);
    if (!qf) return result;

    const ExtractedFeatures* rf = getCachedFeatures(crop.image);  // no mask for reference crops
    if (!rf) return result;

    // Mutual nearest-neighbor + cosine-similarity threshold -- NOT Lowe's
    // ratio test. Deliberate deviation from OrbRansacMatching's recipe,
    // found necessary empirically: an initial Lowe's-ratio version (mirroring
    // ORB's exact matching logic for direct comparability) found ZERO valid
    // matches across an entire flight, even though extraction itself was
    // confirmed working (150-1024 real keypoints per image, same-domain
    // self-match smoke test: 1024/1024 inliers). Diagnosed directly: cross-
    // domain (drone-vs-satellite) best-match L2 distances cluster around
    // 0.5-1.0 on these unit-normalized descriptors (cosine similarity
    // ~0.5-0.75) -- no keypoint has a distinctively best match, so a ratio
    // test comparing best-vs-second-best is the wrong tool here (it assumes
    // a clear winner exists). XFeat's own reference implementation
    // (modules/xfeat.py's match()/batch_match()) uses exactly this mutual-
    // NN + absolute-cosine-threshold strategy instead, with a default
    // min_cossim=0.82 -- used verbatim here, not independently tuned, since
    // this is "use XFeat as intended" not a new hyperparameter search.
    // crossCheck=true implements mutual NN directly (a match only counts if
    // each descriptor is the other's best match in both directions).
    cv::BFMatcher matcher(cv::NORM_L2, /*crossCheck=*/true);
    std::vector<cv::DMatch> good;
    try {
        std::vector<cv::DMatch> matches;
        matcher.match(qf->descriptors, rf->descriptors, matches);
        // L2 distance <-> cosine similarity for unit vectors: d^2 = 2(1-c).
        constexpr float kMinCosSim = 0.82f;
        const float kMaxDist = std::sqrt(2.0f * (1.0f - kMinCosSim));
        good.reserve(matches.size());
        for (const auto& m : matches)
            if (m.distance < kMaxDist)
                good.push_back(m);
    } catch (...) {
        return result;
    }

    if (static_cast<int>(good.size()) < 8) return result;

    std::vector<cv::Point2f> src_pts, dst_pts;
    src_pts.reserve(good.size());
    dst_pts.reserve(good.size());
    for (const auto& m : good) {
        src_pts.push_back(qf->keypoints[m.queryIdx]);
        dst_pts.push_back(rf->keypoints[m.trainIdx]);
    }

    cv::Mat mask;
    cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 5.0, mask);
    if (H.empty()) return result;

    int inliers = cv::countNonZero(mask);
    if (inliers == 0) return result;

    result.homography = H;
    result.inliers = inliers;
    return result;
}
