#include "HybridEstimator.hpp"
#include <cmath>
#include <iostream>

HybridEstimator::HybridEstimator()
    : template_estimator_(), orb_estimator_(500), sift_estimator_(0)
{
}

PositionEstimate HybridEstimator::estimatePosition(
    const cv::Mat& drone_view,
    const std::vector<ReferenceCrop>& reference_crops,
    const std::pair<double, double>& last_position)
{
    if (drone_view.empty() || reference_crops.empty()) {
        return PositionEstimate(last_position, 0.0, -1);
    }

    // Get position estimates from all three methods
    PositionEstimate template_result = template_estimator_.estimatePosition(
        drone_view, reference_crops, last_position);

    PositionEstimate orb_result = orb_estimator_.estimatePosition(
        drone_view, reference_crops, last_position);

    PositionEstimate sift_result = sift_estimator_.estimatePosition(
        drone_view, reference_crops, last_position);

    // ==================== DECISION LOGIC ====================
    
    // 1. If template matching has very high confidence, trust it immediately
    if (template_result.confidence > 0.85) {
        return template_result;
    }

    // 2. Check for consensus among all three methods (very strong signal)
    bool all_agree = (template_result.best_match_idx == orb_result.best_match_idx &&
                      orb_result.best_match_idx == sift_result.best_match_idx &&
                      template_result.best_match_idx != -1);
    
    if (all_agree) {
        // All three methods agree - very high confidence
        double avg_confidence = (template_result.confidence + 
                                orb_result.confidence + 
                                sift_result.confidence) / 3.0;
        
        PositionEstimate result = template_result;
        result.confidence = std::min(0.98, avg_confidence + 0.15);  // Boost confidence
        
        std::cout << "✓ All methods agree on match " << result.best_match_idx 
                  << " (confidence: " << result.confidence << ")" << std::endl;
        return result;
    }

    // 3. Check for two-method consensus
    int template_orb_match = (template_result.best_match_idx == orb_result.best_match_idx && 
                             template_result.best_match_idx != -1) ? 1 : 0;
    int template_sift_match = (template_result.best_match_idx == sift_result.best_match_idx && 
                              template_result.best_match_idx != -1) ? 1 : 0;
    int orb_sift_match = (orb_result.best_match_idx == sift_result.best_match_idx && 
                         orb_result.best_match_idx != -1) ? 1 : 0;
    
    // Template + ORB agree with good confidence
    if (template_orb_match && template_result.confidence > 0.55 && orb_result.confidence > 0.55) {
        PositionEstimate result = template_result;
        result.confidence = std::min(0.95, (template_result.confidence + orb_result.confidence) / 2.0 + 0.1);
        return result;
    }
    
    // Template + SIFT agree with good confidence
    if (template_sift_match && template_result.confidence > 0.55 && sift_result.confidence > 0.55) {
        PositionEstimate result = template_result;
        result.confidence = std::min(0.95, (template_result.confidence + sift_result.confidence) / 2.0 + 0.1);
        return result;
    }
    
    // ORB + SIFT agree (feature-based consensus)
    if (orb_sift_match && orb_result.confidence > 0.50 && sift_result.confidence > 0.50) {
        // Average the feature-based results
        PositionEstimate result = orb_result;
        result.confidence = std::min(0.92, (orb_result.confidence + sift_result.confidence) / 2.0 + 0.08);
        return result;
    }

    // 4. Individual method selection based on confidence thresholds
    
    // SIFT has very high confidence (SIFT is generally more reliable than ORB)
    if (sift_result.confidence > 0.75) {
        return sift_result;
    }
    
    // ORB has high confidence and others are weak
    if (orb_result.confidence > 0.70 && template_result.confidence < 0.5) {
        return orb_result;
    }
    
    // Template has good confidence
    if (template_result.confidence > 0.60) {
        return template_result;
    }
    
    // SIFT has moderate confidence
    if (sift_result.confidence > 0.55) {
        return sift_result;
    }
    
    // ORB has moderate confidence
    if (orb_result.confidence > 0.50) {
        return orb_result;
    }
    
    // Template has moderate confidence
    if (template_result.confidence > 0.40) {
        return template_result;
    }

    // 5. Last resort: Select best of the three based on confidence
    PositionEstimate best_result = template_result;
    
    if (orb_result.confidence > best_result.confidence) {
        best_result = orb_result;
    }
    
    if (sift_result.confidence > best_result.confidence) {
        best_result = sift_result;
    }
    
    // If all methods have very low confidence, log a warning
    if (best_result.confidence < 0.3) {
        std::cout << "⚠️  Warning: All methods have low confidence (best: " 
                  << best_result.confidence << ")" << std::endl;
    }
    
    return best_result;
}