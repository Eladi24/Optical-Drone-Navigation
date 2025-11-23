#include "KalmanFilter.hpp"
#include <cmath>

DroneKalmanFilter::DroneKalmanFilter(double process_noise, double measurement_noise)
    : initialized_(false), process_noise_(process_noise), measurement_noise_(measurement_noise)
{
    // State: [lat, lng, vel_lat, vel_lng]
    kf_.init(4, 2, 0);
    
    // Transition matrix (constant velocity model)
    // x_k = F * x_{k-1}
    kf_.transitionMatrix = (cv::Mat_<float>(4, 4) << 
        1, 0, 1, 0,   // lat += vel_lat * dt
        0, 1, 0, 1,   // lng += vel_lng * dt
        0, 0, 1, 0,   // vel_lat (constant)
        0, 0, 0, 1);  // vel_lng (constant)
    
    // Measurement matrix (we only measure position)
    kf_.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);
    
    // Process noise covariance
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(process_noise));
    
    // Measurement noise covariance (will be adjusted by confidence)
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(measurement_noise));
    
    // Error covariance
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));
}

void DroneKalmanFilter::initialize(double lat, double lng, double heading, double speed) {
    kf_.statePost.at<float>(0) = lat;
    kf_.statePost.at<float>(1) = lng;
    
    // Convert heading and speed to velocity components
    double heading_rad = heading * M_PI / 180.0;
    kf_.statePost.at<float>(2) = speed * std::cos(heading_rad) / 111320.0;  // ~meters_per_degree_lat
    kf_.statePost.at<float>(3) = speed * std::sin(heading_rad) / 111320.0;
    
    initialized_ = true;
}

std::pair<double, double> DroneKalmanFilter::predict(double dt) {
    // Update transition matrix with actual dt
    kf_.transitionMatrix.at<float>(0, 2) = dt;
    kf_.transitionMatrix.at<float>(1, 3) = dt;
    
    cv::Mat prediction = kf_.predict();
    return {prediction.at<float>(0), prediction.at<float>(1)};
}

std::pair<double, double> DroneKalmanFilter::update(
    double measured_lat, double measured_lng, 
    double confidence, double innovation_threshold)
{
    cv::Mat measurement = (cv::Mat_<float>(2, 1) << measured_lat, measured_lng);
    
    // Calculate innovation (measurement - prediction)
    double innovation = getInnovation(measured_lat, measured_lng);
    
    // Adaptive measurement noise based on confidence
    // Low confidence → high noise → less trust in measurement
    double adaptive_noise = measurement_noise_ / (confidence + 0.1);
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(adaptive_noise));
    
    // Outlier rejection: if innovation is too large, don't update
    if (innovation > innovation_threshold && confidence < 0.7) {
        std::cout << "⚠️ Kalman: Rejecting outlier (innovation=" << innovation 
                  << "m, conf=" << confidence << ")" << std::endl;
        return getPosition();  // Return predicted position
    }
    
    // Update with measurement
    cv::Mat corrected = kf_.correct(measurement);
    return {corrected.at<float>(0), corrected.at<float>(1)};
}

std::pair<double, double> DroneKalmanFilter::getPosition() const {
    return {kf_.statePost.at<float>(0), kf_.statePost.at<float>(1)};
}

double DroneKalmanFilter::getInnovation(double measured_lat, double measured_lng) const {
    double pred_lat = kf_.statePre.at<float>(0);
    double pred_lng = kf_.statePre.at<float>(1);
    
    // Approximate distance (works for small areas)
    double dlat = (measured_lat - pred_lat) * 111320.0;
    double dlng = (measured_lng - pred_lng) * 111320.0;
    return std::sqrt(dlat * dlat + dlng * dlng);
}

void DroneKalmanFilter::updateHeading(double new_heading) {
    if (!initialized_) return;
    
    // Calculate heading change
    double heading_change = std::abs(new_heading - last_heading_);
    if (heading_change > 180.0) heading_change = 360.0 - heading_change;
    
    // If significant turn, increase process noise temporarily
    if (heading_change > 15.0) {
        double increased_noise = process_noise_ * 5.0;  // 5x during turns
        cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(increased_noise));
    } else {
        cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(process_noise_));
    }
    
    last_heading_ = new_heading;
}

void DroneKalmanFilter::setProcessNoise(double noise) {
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(noise));
}