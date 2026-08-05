#include "ParticleFilter.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace {
constexpr double kMetersPerDegLat = 111320.0;

// Planar approximation, consistent with CoordinateUtils::calculateDistance
// (valid for the small areas a single flight covers).
double meters_per_deg_lng(double lat_deg) {
    return kMetersPerDegLat * std::cos(lat_deg * M_PI / 180.0);
}

double distance_m(double lat1, double lng1, double lat2, double lng2, double mplng) {
    double dlat = (lat2 - lat1) * kMetersPerDegLat;
    double dlng = (lng2 - lng1) * mplng;
    return std::sqrt(dlat * dlat + dlng * dlng);
}
}  // namespace

ParticleFilter::ParticleFilter(int num_particles)
    : num_particles_(num_particles), rng_(std::random_device{}())
{
}

void ParticleFilter::initialize(double lat, double lng, double heading, double speed) {
    double mplng = meters_per_deg_lng(lat);
    double heading_rad = heading * M_PI / 180.0;
    double vel_lat = speed * std::cos(heading_rad) / kMetersPerDegLat;
    double vel_lng = speed * std::sin(heading_rad) / mplng;

    // Small initial spread (~15m) around the seeded start position -- the
    // start itself is a real GPS-verified point (see readGroundTruthStart),
    // not something to hedge against with a wide initial cloud.
    std::normal_distribution<double> pos_noise(0.0, 15.0);

    particles_.clear();
    particles_.reserve(num_particles_);
    for (int i = 0; i < num_particles_; ++i) {
        double lat_off = pos_noise(rng_) / kMetersPerDegLat;
        double lng_off = pos_noise(rng_) / mplng;
        particles_.push_back({lat + lat_off, lng + lng_off, vel_lat, vel_lng,
                               1.0 / num_particles_});
    }
    initialized_ = true;
}

std::pair<double, double> ParticleFilter::predict(double dt) {
    // Process noise: small random-walk perturbation to both position and
    // velocity each step, so uncertainty grows between observations instead
    // of particles marching in perfect lockstep off a single stale velocity
    // estimate. ~5 m/s^2-ish velocity noise, tunable -- see CLAUDE.md
    // Investigation Log's tunables list for this filter.
    std::normal_distribution<double> vel_noise(0.0, 3.0);   // m/s per step
    std::normal_distribution<double> pos_noise(0.0, 2.0);   // m per step, direct jitter

    for (auto& p : particles_) {
        double mplng = meters_per_deg_lng(p.lat);
        p.vel_lat += vel_noise(rng_) / kMetersPerDegLat;
        p.vel_lng += vel_noise(rng_) / mplng;
        p.lat += p.vel_lat * dt + pos_noise(rng_) / kMetersPerDegLat;
        p.lng += p.vel_lng * dt + pos_noise(rng_) / mplng;
    }
    return getPosition();
}

std::pair<double, double> ParticleFilter::update(
    const std::vector<std::pair<std::pair<double, double>, double>>& candidates,
    double confidence)
{
    if (candidates.empty()) {
        // No usable observation this frame (e.g. ORB's early-return path
        // for too few keypoints) -- leave weights as propagated, no update.
        return getPosition();
    }

    // Wider (more forgiving) likelihood when confidence is low, mirroring
    // DroneKalmanFilter::update's adaptive_noise = measurement_noise_ /
    // (confidence + 0.1). base_sigma is the typical match-uncertainty scale
    // at confidence ~1.0 -- tunable, not empirically calibrated yet.
    constexpr double kBaseSigmaM = 100.0;
    double sigma = kBaseSigmaM / (confidence + 0.1);

    double weight_sum = 0.0;
    for (auto& p : particles_) {
        double mplng = meters_per_deg_lng(p.lat);
        double best_likelihood = 0.0;
        for (const auto& cand : candidates) {
            double dist = distance_m(p.lat, p.lng, cand.first.first, cand.first.second, mplng);
            double likelihood = std::exp(-0.5 * (dist / sigma) * (dist / sigma));
            best_likelihood = std::max(best_likelihood, likelihood);
        }
        // Small floor so a particle far from every candidate doesn't hit
        // exactly zero weight and become permanently unrecoverable outside
        // of resampling's diversity injection.
        p.weight *= std::max(best_likelihood, 1e-6);
        weight_sum += p.weight;
    }

    if (weight_sum > 0.0) {
        for (auto& p : particles_) p.weight /= weight_sum;
    }

    resampleIfNeeded(candidates);
    return getPosition();
}

void ParticleFilter::resampleIfNeeded(
    const std::vector<std::pair<std::pair<double, double>, double>>& candidates)
{
    double sum_sq = 0.0;
    for (const auto& p : particles_) sum_sq += p.weight * p.weight;
    double effective_sample_size = (sum_sq > 0.0) ? (1.0 / sum_sq) : 0.0;

    if (effective_sample_size >= num_particles_ / 2.0) return;

    // Diversity fraction: seeded near this frame's actual observed
    // candidates (not randomly across the whole map, and not ground truth)
    // -- gives the filter a real, principled chance to pick up a candidate
    // the current population has drifted away from. This is the native
    // equivalent of Phase A2's hand-rolled "recovery mode", but a graceful
    // partial reseed rather than an all-or-nothing full-database fallback.
    // Lowered from 0.05 after the first live measurement (flight 01) showed
    // a heavier error tail than the Kalman baseline (max 7377m vs 5982m) --
    // suspected contributor alongside the getPosition() density fix above.
    // See CLAUDE.md Investigation Log.
    constexpr double kDiversityFraction = 0.02;
    int num_diversity = static_cast<int>(num_particles_ * kDiversityFraction);

    // Systematic resampling: low-variance, standard choice over naive
    // multinomial resampling.
    std::vector<Particle> resampled;
    resampled.reserve(num_particles_);

    std::vector<double> cumulative(particles_.size());
    double running = 0.0;
    for (size_t i = 0; i < particles_.size(); ++i) {
        running += particles_[i].weight;
        cumulative[i] = running;
    }

    std::uniform_real_distribution<double> start_dist(0.0, 1.0 / num_particles_);
    double u = start_dist(rng_);
    size_t idx = 0;
    int num_main = num_particles_ - num_diversity;
    for (int i = 0; i < num_main; ++i) {
        double target = u + static_cast<double>(i) / num_particles_;
        while (idx < cumulative.size() - 1 && cumulative[idx] < target) ++idx;
        Particle p = particles_[idx];
        p.weight = 1.0 / num_particles_;
        resampled.push_back(p);
    }

    std::normal_distribution<double> diversity_noise(0.0, 50.0);  // meters
    std::uniform_int_distribution<size_t> cand_pick(0, candidates.empty() ? 0 : candidates.size() - 1);
    for (int i = 0; i < num_diversity; ++i) {
        double lat, lng;
        if (!candidates.empty()) {
            auto& c = candidates[cand_pick(rng_)];
            lat = c.first.first;
            lng = c.first.second;
        } else {
            lat = particles_[idx].lat;
            lng = particles_[idx].lng;
        }
        double mplng = meters_per_deg_lng(lat);
        lat += diversity_noise(rng_) / kMetersPerDegLat;
        lng += diversity_noise(rng_) / mplng;
        resampled.push_back({lat, lng, 0.0, 0.0, 1.0 / num_particles_});
    }

    particles_ = std::move(resampled);
}

std::pair<double, double> ParticleFilter::getPosition() const {
    if (particles_.empty()) return {0.0, 0.0};

    // Highest-weight particle anchors the cluster; average nearby particles
    // (within ~100m) around it for a smoother estimate than a single noisy
    // point, without pulling in a separate, distant hypothesis if the
    // population is genuinely bimodal.
    //
    // The anchor is picked by LOCAL DENSITY (sum of neighbor weights within
    // 100m), not a single particle's own weight. Right after a resample
    // every particle's individual weight is briefly uniform (1/N), so
    // "highest own weight" can't yet distinguish the main, well-supported
    // cluster from a handful of just-seeded diversity particles -- but
    // local density still can, because resampling already drew particle
    // *count* proportional to prior weight, so the main cluster is denser
    // by construction even when everyone's current weight looks the same.
    // (Found and fixed after the first live measurement showed a heavier
    // error tail than the Kalman baseline -- see CLAUDE.md Investigation
    // Log.)
    const Particle* anchor = &particles_[0];
    double best_density = -1.0;
    for (const auto& p : particles_) {
        double mplng_p = meters_per_deg_lng(p.lat);
        double density = 0.0;
        for (const auto& q : particles_)
            if (distance_m(p.lat, p.lng, q.lat, q.lng, mplng_p) <= 100.0)
                density += q.weight;
        if (density > best_density) {
            best_density = density;
            anchor = &p;
        }
    }

    double mplng = meters_per_deg_lng(anchor->lat);
    double sum_lat = 0.0, sum_lng = 0.0, sum_w = 0.0;
    for (const auto& p : particles_) {
        if (distance_m(p.lat, p.lng, anchor->lat, anchor->lng, mplng) <= 100.0) {
            sum_lat += p.lat * p.weight;
            sum_lng += p.lng * p.weight;
            sum_w   += p.weight;
        }
    }
    if (sum_w <= 0.0) return {anchor->lat, anchor->lng};
    return {sum_lat / sum_w, sum_lng / sum_w};
}
