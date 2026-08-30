#include "DinoDenseMatching.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
// Must match scripts/export_retrieval_backbone.py --dense (INPUT_SIZE 224,
// DINOv2-S patch14 -> 16x16 = 256 tokens, 384-dim).
constexpr int kInputSize = 224;
constexpr int kGrid      = 16;
constexpr int kNumTokens = kGrid * kGrid;  // 256
constexpr int kDescDim   = 384;
constexpr float kPatchPx = static_cast<float>(kInputSize) / kGrid;  // 14.0 in 224-space

// DINOv2 expects ImageNet-normalized RGB (same as OnnxDeitRetrieval) -- unlike
// XFeat, whose first op grayscales so channel stats don't matter.
constexpr float kImageNetMean[3] = {0.485f, 0.456f, 0.406f};  // RGB order
constexpr float kImageNetStd[3]  = {0.229f, 0.224f, 0.225f};

// A dense 16x16 grid always yields ~256 "keypoints", so extraction validity is
// not a keypoint-count question the way it is for ORB/XFeat -- the real filter
// is on the number of surviving mutual-NN correspondences (below). Only reject
// extraction outright if a mask wiped out almost the whole frame.
constexpr int kMinGridPts = 32;

// Minimum surviving mutual-NN + cosine-threshold matches before RANSAC is even
// attempted, and minimum RANSAC inliers to call the match real. The prototype
// (scripts/proto_dino_dense_matching.py) used 8 as a floor; 12 pre-RANSAC is a
// mild tightening to lean against wrong-crop matches that scrape past with a
// handful of coincidental correspondences. Not independently swept -- a
// reasonable first value, same convention as this project's other first-pass
// thresholds (Step 1 Tukey constant, etc.).
constexpr int kMinGoodMatches = 12;
constexpr int kMinInliers     = 8;

Ort::SessionOptions makeSessionOptions()
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);  // this codebase parallelizes across frames/candidates itself
    return opts;
}
}  // namespace

DinoDenseMatching::DinoDenseMatching(const std::string& model_path, float cos_threshold)
    : env_(ORT_LOGGING_LEVEL_WARNING, "DinoDenseMatching")
    , session_(env_, model_path.c_str(), makeSessionOptions())
    , mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , cos_threshold_(cos_threshold)
{
}

DinoDenseMatching::DenseFeatures DinoDenseMatching::extract(const cv::Mat& image, const cv::Mat& mask)
{
    DenseFeatures result;
    if (image.empty()) return result;

    cv::Mat working = image;
    if (!mask.empty() && mask.size() == image.size()) {
        working = image.clone();
        working.setTo(cv::Scalar(0, 0, 0), mask == 0);
    }

    cv::Mat resized;
    cv::resize(working, resized, cv::Size(kInputSize, kInputSize));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<float> input(static_cast<size_t>(3) * kInputSize * kInputSize);
    const int plane = kInputSize * kInputSize;
    for (int y = 0; y < kInputSize; ++y) {
        const cv::Vec3f* row = rgb.ptr<cv::Vec3f>(y);
        for (int x = 0; x < kInputSize; ++x)
            for (int c = 0; c < 3; ++c)
                input[static_cast<size_t>(c) * plane + y * kInputSize + x] =
                    (row[x][c] - kImageNetMean[c]) / kImageNetStd[c];
    }

    std::array<int64_t, 4> shape{1, 3, kInputSize, kInputSize};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_, input.data(), input.size(), shape.data(), shape.size());

    const char* input_names[]  = {"image"};
    const char* output_names[] = {"patch_tokens"};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                 output_names, 1);

    const float* tok = outputs[0].GetTensorData<float>();  // [1, 256, 384], row-major token order
    size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (count < static_cast<size_t>(kNumTokens) * kDescDim) return result;

    // Patch-grid centres are kept in the 224 RESIZE space (identical for every
    // image), NOT rescaled to original pixels here. Both the mutual-NN match and
    // the RANSAC homography are solved in that symmetric 224x224 space -- exactly
    // as the validated Python prototype does -- then match() composes the result
    // with the two images' resize scales to produce a frame-px -> crop-px
    // homography. Solving in original-pixel space instead is asymmetric (a
    // ~4000px frame vs a ~250px crop) and makes a fixed RANSAC pixel threshold
    // meaningless -- the first C++ version did that and found zero inliers
    // flight-wide despite 20-70 good correspondences per pair.
    const float mx_scale = static_cast<float>(image.cols) / kInputSize;  // 224-space -> original px
    const float my_scale = static_cast<float>(image.rows) / kInputSize;

    std::vector<float> desc_rows;
    desc_rows.reserve(static_cast<size_t>(kNumTokens) * kDescDim);
    result.grid_pts.reserve(kNumTokens);

    const bool have_mask = !mask.empty() && mask.size() == image.size();
    for (int i = 0; i < kNumTokens; ++i) {
        int gr = i / kGrid, gc = i % kGrid;
        cv::Point2f pt224((gc + 0.5f) * kPatchPx, (gr + 0.5f) * kPatchPx);

        if (have_mask) {
            int mx = cv::borderInterpolate(static_cast<int>(pt224.x * mx_scale),
                                           mask.cols, cv::BORDER_REPLICATE);
            int my = cv::borderInterpolate(static_cast<int>(pt224.y * my_scale),
                                           mask.rows, cv::BORDER_REPLICATE);
            if (mask.at<uchar>(my, mx) == 0) continue;
        }

        const float* d = tok + static_cast<size_t>(i) * kDescDim;
        // L2-normalize this token so a plain L2 BFMatcher distance maps to
        // cosine similarity via d^2 = 2(1 - cos) for unit vectors.
        double n = 0.0;
        for (int c = 0; c < kDescDim; ++c) n += static_cast<double>(d[c]) * d[c];
        n = std::sqrt(n);
        float inv = (n > 1e-12) ? static_cast<float>(1.0 / n) : 0.0f;
        for (int c = 0; c < kDescDim; ++c) desc_rows.push_back(d[c] * inv);
        result.grid_pts.push_back(pt224);   // 224-space; match() rescales the final homography
    }

    if (static_cast<int>(result.grid_pts.size()) >= kMinGridPts) {
        result.descriptors = cv::Mat(static_cast<int>(result.grid_pts.size()), kDescDim, CV_32F,
                                      desc_rows.data()).clone();
        result.valid = true;
    }
    return result;
}

const DinoDenseMatching::DenseFeatures* DinoDenseMatching::getCachedFeatures(const cv::Mat& image,
                                                                             const cv::Mat& mask)
{
    const uchar* key = image.data;
    auto it = feature_cache_.find(key);
    if (it != feature_cache_.end())
        return it->second.valid ? &it->second : nullptr;

    auto [ins, _] = feature_cache_.emplace(key, extract(image, mask));
    return ins->second.valid ? &ins->second : nullptr;
}

MatchResult DinoDenseMatching::match(const cv::Mat& frame, const ReferenceCrop& crop)
{
    MatchResult result;
    if (frame.empty() || crop.image.empty()) return result;

    const DenseFeatures* qf = getCachedFeatures(frame, feature_mask_);
    if (!qf) return result;
    const DenseFeatures* rf = getCachedFeatures(crop.image);  // no mask for reference crops
    if (!rf) return result;

    // Mutual nearest-neighbour (crossCheck=true) + an ABSOLUTE cosine threshold
    // -- same rationale as XFeatMatching: cross-domain best-match distances on
    // these unit-normalized learned descriptors are too uniform for a Lowe's
    // ratio test, and a cosine *ratio* test specifically fails on DINOv2 patch
    // tokens because they're locally smooth (adjacent patches near-identical
    // cosine, best-minus-second always tiny) -- verified in the Python
    // prototype, documented in scripts/proto_dino_dense_matching.py.
    cv::BFMatcher matcher(cv::NORM_L2, /*crossCheck=*/true);
    std::vector<cv::Point2f> src_pts, dst_pts;
    try {
        std::vector<cv::DMatch> matches;
        matcher.match(qf->descriptors, rf->descriptors, matches);
        const float max_dist = std::sqrt(2.0f * (1.0f - cos_threshold_));
        for (const auto& m : matches) {
            if (m.distance >= max_dist) continue;
            src_pts.push_back(qf->grid_pts[m.queryIdx]);
            dst_pts.push_back(rf->grid_pts[m.trainIdx]);
        }
    } catch (...) {
        return result;
    }

    if (static_cast<int>(src_pts.size()) < kMinGoodMatches) return result;

    // src_pts / dst_pts are both in the symmetric 224x224 space -- solve the
    // homography there, then compose it with the resize scales below. RANSAC
    // reprojection threshold is ONE patch cell (14 px): the correspondences are
    // quantized to the 16x16 token grid, so even a geometrically perfect dense
    // match carries up to +-7 px per axis of grid-snap error -- a tighter
    // threshold (the prototype's 5.0) rejects correct correspondences as
    // outliers, which is why the first C++ build found zero inliers flight-wide.
    cv::Mat mask;
    cv::Mat H224 = cv::findHomography(src_pts, dst_pts, cv::RANSAC, kPatchPx, mask);
    if (H224.empty()) return result;

    int inliers = cv::countNonZero(mask);
    if (std::getenv("DINO_DEBUG"))
        fprintf(stderr, "[dino] good=%zu inliers=%d\n", src_pts.size(), inliers);
    if (inliers < kMinInliers) return result;

    // ... then compose with the two images' resize scales so the returned
    // homography maps ORIGINAL frame pixels -> ORIGINAL crop pixels, which is
    // the contract SplitPipelineEstimator's perspectiveTransform expects.
    //   H_orig = Sc_inv * H224 * Sf
    //   Sf     : frame_px  -> 224   = diag(224/fw, 224/fh, 1)
    //   Sc_inv : 224       -> crop_px = diag(cw/224, ch/224, 1)
    const double fw = frame.cols, fh = frame.rows;
    const double cw = crop.image.cols, ch = crop.image.rows;
    cv::Matx33d Sf(kInputSize / fw, 0, 0,
                   0, kInputSize / fh, 0,
                   0, 0, 1);
    cv::Matx33d ScInv(cw / kInputSize, 0, 0,
                      0, ch / kInputSize, 0,
                      0, 0, 1);
    cv::Mat H = cv::Mat(ScInv * cv::Matx33d(H224.ptr<double>()) * Sf);

    result.homography = H;
    result.inliers = inliers;
    return result;
}
