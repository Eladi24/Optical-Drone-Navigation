#include "KalmanFilter.hpp"
#include <cmath>
#include <iostream>

namespace {
// Writes process_noise_meters as the correct per-axis degrees^2 variance
// into the diagonal of a 4x4 [lat, lng, vel_lat, vel_lng] covariance matrix.
// Both position and velocity components use the same per-axis scale --
// preserves the original code's "one magnitude for the whole matrix" design,
// only fixing which conversion factor applies to which axis.
void setAnisotropicNoise(cv::Mat& cov, double noise_meters,
                          double meters_per_degree_lat, double meters_per_degree_lng) {
    cov = cv::Mat::zeros(cov.rows, cov.cols, CV_64F);
    double n_lat = noise_meters / (meters_per_degree_lat * meters_per_degree_lat);
    double n_lng = noise_meters / (meters_per_degree_lng * meters_per_degree_lng);
    cov.at<double>(0, 0) = n_lat;
    cov.at<double>(1, 1) = n_lng;
    if (cov.rows > 2) {
        cov.at<double>(2, 2) = n_lat;
        cov.at<double>(3, 3) = n_lng;
    }
}
}  // namespace

DroneKalmanFilter::DroneKalmanFilter(double process_noise, double measurement_noise,
                                      double meters_per_degree_lat, double meters_per_degree_lng)
    : initialized_(false), process_noise_(process_noise), measurement_noise_(measurement_noise)
    , meters_per_degree_lat_(meters_per_degree_lat), meters_per_degree_lng_(meters_per_degree_lng)
{
    // State: [lat, lng, vel_lat, vel_lng]
    kf_.init(4, 2, 0, CV_64F);

    // Transition matrix (constant velocity model)
    // x_k = F * x_{k-1}
    kf_.transitionMatrix = (cv::Mat_<double>(4, 4) <<
        1, 0, 1, 0,   // lat += vel_lat * dt
        0, 1, 0, 1,   // lng += vel_lng * dt
        0, 0, 1, 0,   // vel_lat (constant)
        0, 0, 0, 1);  // vel_lng (constant)

    // Measurement matrix (we only measure position)
    kf_.measurementMatrix = (cv::Mat_<double>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    // Process noise covariance -- per-axis, not isotropic (see setAnisotropicNoise)
    setAnisotropicNoise(kf_.processNoiseCov, process_noise, meters_per_degree_lat_, meters_per_degree_lng_);

    // Measurement noise covariance (will be adjusted by confidence)
    setAnisotropicNoise(kf_.measurementNoiseCov, measurement_noise, meters_per_degree_lat_, meters_per_degree_lng_);

    // Error covariance
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));
}

void DroneKalmanFilter::initialize(double lat, double lng, double heading, double speed) {
    kf_.statePost.at<double>(0) = lat;
    kf_.statePost.at<double>(1) = lng;

    // Convert heading and speed to velocity components
    double heading_rad = heading * M_PI / 180.0;
    kf_.statePost.at<double>(2) = speed * std::cos(heading_rad) / meters_per_degree_lat_;
    kf_.statePost.at<double>(3) = speed * std::sin(heading_rad) / meters_per_degree_lng_;

    initialized_ = true;
}

std::pair<double, double> DroneKalmanFilter::predict(double dt) {
    // Update transition matrix with actual dt
    kf_.transitionMatrix.at<double>(0, 2) = dt;
    kf_.transitionMatrix.at<double>(1, 3) = dt;

    cv::Mat prediction = kf_.predict();
    return {prediction.at<double>(0), prediction.at<double>(1)};
}

std::pair<double, double> DroneKalmanFilter::update(
    double measured_lat, double measured_lng,
    double confidence, double innovation_threshold)
{
    cv::Mat measurement = (cv::Mat_<double>(2, 1) << measured_lat, measured_lng);

    // Calculate innovation (measurement - prediction)
    double innovation = getInnovation(measured_lat, measured_lng);

    // Adaptive measurement noise based on confidence
    // Low confidence → high noise → less trust in measurement
    double adaptive_noise = measurement_noise_ / (confidence + 0.1);
    setAnisotropicNoise(kf_.measurementNoiseCov, adaptive_noise, meters_per_degree_lat_, meters_per_degree_lng_);

    // Outlier rejection: if innovation is too large, don't update
    if (innovation > innovation_threshold && confidence < 0.7) {
        std::cout << "⚠️ Kalman: Rejecting outlier (innovation=" << innovation
                  << "m, conf=" << confidence << ")" << std::endl;
        return getPosition();  // Return predicted position
    }

    // Update with measurement
    cv::Mat corrected = kf_.correct(measurement);
    return {corrected.at<double>(0), corrected.at<double>(1)};
}

std::pair<double, double> DroneKalmanFilter::getPosition() const {
    return {kf_.statePost.at<double>(0), kf_.statePost.at<double>(1)};
}

double DroneKalmanFilter::getInnovation(double measured_lat, double measured_lng) const {
    double pred_lat = kf_.statePre.at<double>(0);
    double pred_lng = kf_.statePre.at<double>(1);

    // Approximate distance (works for small areas)
    double dlat = (measured_lat - pred_lat) * meters_per_degree_lat_;
    double dlng = (measured_lng - pred_lng) * meters_per_degree_lng_;
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
        setAnisotropicNoise(kf_.processNoiseCov, increased_noise, meters_per_degree_lat_, meters_per_degree_lng_);
    } else {
        setAnisotropicNoise(kf_.processNoiseCov, process_noise_, meters_per_degree_lat_, meters_per_degree_lng_);
    }

    last_heading_ = new_heading;
}

void DroneKalmanFilter::setProcessNoise(double noise) {
    setAnisotropicNoise(kf_.processNoiseCov, noise, meters_per_degree_lat_, meters_per_degree_lng_);
}