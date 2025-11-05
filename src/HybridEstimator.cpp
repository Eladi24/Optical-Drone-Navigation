#include "HybridEstimator.hpp"
#include <cmath>

HybridEstimator::HybridEstimator()
    : template_estimator_(), orb_estimator_(500)
{
}

PositionEstimate HybridEstimator::estimatePosition(
    const cv::Mat& drone_view,
    const std::vector<std::pair<double, double>>& ref_crop_coords,
    const std::unordered_map<std::pair<double, double>, cv::Mat, CoordinateHash>& reference_crops,
    const std::pair<double, double>& last_position)
{
    if (drone_view.empty() || ref_crop_coords.empty()) {
        return PositionEstimate(last_position, 0.0, -1);
    }

    // Get position estimate using template matching
    PositionEstimate template_result = template_estimator_.estimatePosition(
        drone_view, ref_crop_coords, reference_crops, last_position);

    // Get position estimate using feature matching
    PositionEstimate orb_result = orb_estimator_.estimatePosition(
        drone_view, ref_crop_coords, reference_crops, last_position);

    // Decision logic - prioritize methods based on confidence and agreement
    
    // If template matching is very confident, use it
    if (template_result.confidence > 0.8) {
        return template_result;
    }

    // If both methods agree with reasonable confidence, boost confidence
    if (template_result.confidence > 0.6 && 
        template_result.best_match_idx == orb_result.best_match_idx &&
        template_result.best_match_idx != -1) {
        
        PositionEstimate result = template_result;
        result.confidence = std::min(0.95, template_result.confidence + 0.1);
        return result;
    }

    // If ORB has good confidence and template is weak, use ORB
    if (orb_result.confidence > 0.6 && template_result.confidence < 0.5) {
        return orb_result;
    }

    // If template has moderate confidence, use it
    if (template_result.confidence > 0.4) {
        return template_result;
    }

    // If ORB has any reasonable result, use it
    if (orb_result.confidence > 0.4) {
        return orb_result;
    }

    // Last resort: return whichever has higher confidence
    if (template_result.confidence >= orb_result.confidence) {
        return template_result;
    } else {
        return orb_result;
    }
}