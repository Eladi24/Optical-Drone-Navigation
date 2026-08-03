#pragma once
#include "PositionEstimation.hpp"
#include "KalmanFilter.hpp"
#include "EdgeProcessor.hpp"
#include <opencv2/video/tracking.hpp>
#include <future>
#include <atomic>

// ---------------------------------------------------------------------------
// OpticalFlowTracker — Lucas-Kanade sparse optical flow position estimator.
//
// Implements IPositionEstimator so it can be swapped in wherever the
// existing ORB/SIFT estimators are used.
//
// Algorithm:
//   • Detect Shi-Tomasi corners in the first frame (masked for HUD).
//   • Each subsequent call tracks them with calcOpticalFlowPyrLK.
//   • estimateAffinePartial2D(RANSAC) converts pixel displacement → metres.
//   • GSD (metres/pixel) is slowly updated from the affine scale component
//     to handle gradual altitude changes.
//   • Every RELOC_INTERVAL frames, an async edge-template re-localisation
//     runs against a local satellite window to correct long-term drift.
// ---------------------------------------------------------------------------
class OpticalFlowTracker : public IPositionEstimator {
public:
    // @param gsd_init          Initial GSD estimate (m/px) from GlobalLocator.
    // @param satellite_map     Full satellite map image (kept by reference —
    //                          must outlive this object).
    // @param center_lat/lng    Map centre geo-coordinates.
    // @param center_x/y        Map centre pixel coordinates.
    // @param mpp               Map metres-per-pixel.
    // @param m_per_deg_lat/lng Degrees-to-metres conversion.
    // @param hud_mask          Mask suppressing HUD pixels (may be empty).
    OpticalFlowTracker(
        double         gsd_init,
        const cv::Mat& satellite_map,
        double center_lat,  double center_lng,
        int    center_x,    int    center_y,
        double mpp,
        double m_per_deg_lat,
        double m_per_deg_lng,
        const cv::Mat& hud_mask = cv::Mat());

    ~OpticalFlowTracker() override;

    // precompute() is a no-op — no crop descriptor cache needed.
    void precompute(const std::vector<ReferenceCrop>& /*crops*/) override {}

    // Core estimation call.
    // drone_view      : current preprocessed (CLAHE, HUD-masked) greyscale frame.
    // reference_crops : unused by this estimator (passed for API compatibility).
    // last_position   : fallback if tracking is lost.
    PositionEstimate estimatePosition(
        const cv::Mat&                    drone_view,
        const std::vector<ReferenceCrop>& reference_crops,
        const std::pair<double, double>&  last_position) override;

    std::string getName() const override { return "optical_flow"; }

    // Set the GSD after initialization (called by VideoProcessing after lock).
    void setGsd(double gsd) { gsd_ = gsd; }

    // Number of active tracked points (useful for diagnostics).
    int trackedPointCount() const { return static_cast<int>(tracked_pts_.size()); }

private:
    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------
    void detectFeatures(const cv::Mat& gray);
    void mergeNewFeatures(const cv::Mat& gray);
    PositionEstimate attemptRelocalization(const cv::Mat& frame_gray,
                                           const std::pair<double, double>& predicted);

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    double gsd_;                        // metres/pixel in drone video
    const cv::Mat& satellite_map_;
    double center_lat_, center_lng_;
    int    center_x_,   center_y_;
    double mpp_;
    double m_per_deg_lat_, m_per_deg_lng_;
    cv::Mat hud_mask_;

    cv::Mat                    prev_gray_;
    std::vector<cv::Point2f>   tracked_pts_;
    std::pair<double, double>  current_pos_;
    bool                       initialized_;

    // LK optical-flow parameters (set once in constructor)
    cv::TermCriteria lk_criteria_;

    // Re-localization
    static const int RELOC_INTERVAL = 60;  // frames between reloc attempts
    int              frames_since_reloc_;
    std::future<PositionEstimate> reloc_future_;

    // Satellite edge map (precomputed once, used for re-localization)
    cv::Mat sat_gray_;
    cv::Mat sat_edges_;
};
