#pragma once
// Minimal per-stage wall-clock profiler, OFF unless the environment variable
// DRONENAV_PROFILE is set (then every ScopedStage costs one steady_clock read on entry and exit;
// otherwise one cached bool check). Added 2026-11-17 to find where the ~2 s/frame of the
// split_* pipelines goes before deciding on speed work (no-video batch mode, caching, GPU).
//
// Thread-safe: worker-thread stages (per-candidate matching) accumulate under a mutex, so their
// totals are SUMMED CPU-side time across threads, not wall time -- the report labels them.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace StageProfiler {

inline bool enabled() {
    static const bool on = std::getenv("DRONENAV_PROFILE") != nullptr;
    return on;
}

struct Acc { double total_ms = 0.0; long count = 0; double max_ms = 0.0; };

inline std::mutex& mu() { static std::mutex m; return m; }
inline std::map<std::string, Acc>& table() { static std::map<std::string, Acc> t; return t; }

inline void add(const std::string& name, double ms) {
    std::lock_guard<std::mutex> lk(mu());
    Acc& a = table()[name];
    a.total_ms += ms; a.count += 1; if (ms > a.max_ms) a.max_ms = ms;
}

class ScopedStage {
public:
    explicit ScopedStage(const char* name) : name_(name), on_(enabled()) {
        if (on_) t0_ = std::chrono::steady_clock::now();
    }
    ~ScopedStage() {
        if (!on_) return;
        auto dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
        add(name_, dt);
    }
private:
    const char* name_;
    bool on_;
    std::chrono::steady_clock::time_point t0_;
};

// Prints the table; `frames` normalises totals to ms/frame.
inline void report(long frames, const char* title) {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lk(mu());
    std::printf("\n[PROFILE] %s -- %ld frames\n", title, frames);
    std::printf("[PROFILE] %-34s %10s %8s %10s %10s\n", "stage", "ms/frame", "calls", "ms/call", "max ms");
    for (const auto& [name, a] : table()) {
        double per_frame = frames > 0 ? a.total_ms / frames : 0.0;
        double per_call  = a.count > 0 ? a.total_ms / a.count : 0.0;
        std::printf("[PROFILE] %-34s %10.1f %8ld %10.1f %10.1f\n", name.c_str(), per_frame, a.count, per_call, a.max_ms);
    }
    std::fflush(stdout);
}

}  // namespace StageProfiler
