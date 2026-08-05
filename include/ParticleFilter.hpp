#pragma once
#include <utility>
#include <vector>
#include <random>

// ---------------------------------------------------------------------------
// Multi-hypothesis position tracker -- alternative to DroneKalmanFilter for
// processVideoNavigation() (see CLAUDE.md Investigation Log: "Particle
// Filter"). Same call shape (initialize/predict/update/getPosition) so it
// drops into the existing loop with minimal disruption; DroneKalmanFilter is
// left completely unmodified and still the default.
//
// Motivation: DroneKalmanFilter maintains one belief and gates every new
// measurement against it (accept or reject) -- measured directly in Phase A2
// (see CLAUDE.md) to trap the filter on a wrong-but-confident belief, since a
// correction that disagrees with it needs to individually clear a high
// confidence bar most raw matches never reach. A particle filter has no
// single belief to defend: many weighted hypotheses coexist, and a
// correction just needs to be one of them, gaining weight as evidence
// accumulates rather than winning a binary test against an incumbent.
//
// Scope note: this does not add new per-frame information -- update() only
// ever sees the same candidate list ORBFeatureEstimator's RANSAC stage
// already computes (see PositionEstimate::candidates). It targets weak,
// intermittent evidence being fused badly over time, not the case where the
// true crop never appears among the candidates at all.
// ---------------------------------------------------------------------------
class ParticleFilter {
public:
    explicit ParticleFilter(int num_particles = 300);

    // Seeds all particles as a small Gaussian cloud around (lat, lng).
    // heading/speed set initial velocity, same convention as
    // DroneKalmanFilter::initialize (heading degrees 0=north CW, speed m/s).
    void initialize(double lat, double lng, double heading, double speed);

    // Propagates every particle with a constant-velocity step plus
    // per-particle random-walk process noise. Returns the same "current
    // best estimate" getPosition() would (interface parity with
    // DroneKalmanFilter::predict(), which also returns its post-propagation
    // state estimate).
    std::pair<double, double> predict(double dt);

    // Reweights particles by likelihood against the nearest of `candidates`
    // (this is what makes the update multi-modal -- see class comment
    // above), confidence-scaled the same way DroneKalmanFilter's adaptive
    // measurement noise is (low confidence -> wider, weaker likelihood).
    // Empty `candidates` (e.g. ORB's early-return "too few keypoints" path,
    // which never populates PositionEstimate::candidates) is treated as "no
    // information this frame" -- weights are left unchanged. Resamples when
    // effective sample size drops below half the particle count, injecting
    // a small fraction of fresh particles near this frame's candidates to
    // preserve the ability to recover (see class comment above). Returns
    // the post-update position estimate.
    std::pair<double, double> update(
        const std::vector<std::pair<std::pair<double, double>, double>>& candidates,
        double confidence);

    // Position of the highest-weight local cluster, not a naive global
    // weighted mean across all particles -- if the population is genuinely
    // bimodal (two live hypotheses), averaging them would produce a
    // position matching neither.
    std::pair<double, double> getPosition() const;

    bool isInitialized() const { return initialized_; }

private:
    struct Particle {
        double lat, lng;
        double vel_lat, vel_lng;  // degrees/sec (same unit convention as DroneKalmanFilter)
        double weight;
    };

    int                  num_particles_;
    std::vector<Particle> particles_;
    bool                 initialized_ = false;
    mutable std::mt19937 rng_;

    void resampleIfNeeded(
        const std::vector<std::pair<std::pair<double, double>, double>>& candidates);
};
