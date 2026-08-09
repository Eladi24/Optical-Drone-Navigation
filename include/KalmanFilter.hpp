#pragma once
#include <opencv2/opencv.hpp>
#include <utility>

class DroneKalmanFilter {
private:
    cv::KalmanFilter kf_;
    bool initialized_;
    double process_noise_;
    double measurement_noise_;
    double last_heading_;  // 🆕 Track heading

    // Local geo-referencing scale, required to convert a metres-scale noise
    // magnitude into the correct per-axis degrees^2 variance (1 degree of
    // longitude != 1 degree of latitude in metres -- see CLAUDE.md's
    // anisotropy findings, same bug class already fixed in
    // generateReferenceCropsGrid()). No default: every caller must supply
    // the run's actual scale rather than silently assuming one.
    double meters_per_degree_lat_;
    double meters_per_degree_lng_;

public:
    DroneKalmanFilter(double process_noise, double measurement_noise,
                       double meters_per_degree_lat, double meters_per_degree_lng);
    
    // Initialize with first position
    void initialize(double lat, double lng, double heading, double speed);
    
    // Predict next state based on motion model
    std::pair<double, double> predict(double dt);
    
    // Update with visual measurement (with confidence-based gating)
    std::pair<double, double> update(double measured_lat, double measured_lng, 
                                     double confidence, double innovation_threshold = 50.0);
    
    // Get current filtered position
    std::pair<double, double> getPosition() const;
    
    // Get innovation (measurement - prediction) for outlier detection
    double getInnovation(double measured_lat, double measured_lng) const;
    
    // 🆕 Update heading and adjust process noise
    void updateHeading(double new_heading);
    
    // 🆕 Increase process noise temporarily
    void setProcessNoise(double noise);
    
    bool isInitialized() const { return initialized_; }
};