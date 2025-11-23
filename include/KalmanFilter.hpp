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
    
public:
    DroneKalmanFilter(double process_noise = 1.0, double measurement_noise = 10.0);
    
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