#pragma once

#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Per-sample configuration for a telemetry-labeled validation dataset (see
// CLAUDE.md's "Strategic Direction" -- validating the pipeline against real
// GPS ground truth before continuing to chase Haifa's content-mismatch
// problem). Mirrors HaifaSamples.hpp's HaifaSampleConfig pattern, but this
// data source brings its own pre-referenced satellite map and per-frame
// telemetry instead of a live Google Maps fetch and no telemetry at all --
// see runDatasetPipelineForSample() in Main.cpp for how the two paths diverge.
//
// Currently populated for UAV-VisLoc (github.com/IntelliSensing/UAV-VisLoc):
// real drone imagery with per-image lat/lng/altitude/pose, matched against a
// high-resolution satellite GeoTIFF, organized as one folder per flight.
// ---------------------------------------------------------------------------
struct DatasetSampleConfig {
    std::string sample_name;     // e.g. "uavvisloc_03" -- keys ground_truth_*.csv etc.
    std::string location_name;   // e.g. "uavvisloc"

    // UAV-VisLoc flight folder layout (see Datasets/UAV_VisLoc_example/README_example.txt):
    std::string image_dir;       // .../<flight>/drone
    std::string image_prefix;    // "<flight>", e.g. "03" -- files are <prefix>_%04d.JPG
    int         frame_count;     // number of drone images in the flight
    std::string telemetry_csv;   // .../<flight>/<flight>.csv

    // Satellite reference map: pre-converted once via
    // scripts/convert_uavvisloc_satellite.py (the raw GeoTIFF is too large to
    // decode directly -- see that script's docstring), geo-referenced against
    // the dataset's own bounding-box CSV.
    std::string satellite_png;   // Images/map_clean_uavvisloc_<flight>.png
    std::string coords_csv;      // .../satellite_ coordinates_range.csv
    std::string map_name;        // lookup key into coords_csv, e.g. "03.tif"

    // Synthetic assembled video (see DatasetVideoAssembler.hpp) -- cached like
    // Haifa's preprocessed video.
    std::string video_path;      // Videos/dataset_<sample_name>.avi
};

// Flight 03 (Taizhou-1): re-enabled as part of STRATEGY.md Phase 0's dev
// split (dev = flights 01 + 03, per STRATEGY.md Sec 6.1). Its map covers
// ~64 km^2 (vs. flight 01's ~20 km^2) -- a much bigger reference-crop
// database to search every frame -- so full-flight runs are noticeably
// slower than flight 01's; that's expected, not a regression. Its own
// pre-fix baseline numbers are already captured in CLAUDE.md's Investigation
// Log for comparison against the Phase 0 fixes.
inline std::vector<DatasetSampleConfig> getFlight03Sample() { return {{
    "uavvisloc_03", "uavvisloc",
    "Datasets/UAV_VisLoc_example/03/drone", "03", 768,
    "Datasets/UAV_VisLoc_example/03/03.csv",
    "Images/map_clean_uavvisloc_03.png",  // ~1.19 m/px -- see flight 01's comment below re: reverting from _fine
    "Datasets/UAV_VisLoc_example/satellite_ coordinates_range.csv",
    "03.tif",
    "Videos/dataset_uavvisloc_03.avi",
}}; }

inline std::vector<DatasetSampleConfig> getDatasetSamples() {
    std::vector<DatasetSampleConfig> samples;

    // Flight 01 (Changjiang-20): second flight tested specifically to
    // separate "flight 03 is content-hard" from "the pipeline is
    // broken" -- see CLAUDE.md Investigation Log. Content is a river/
    // dockside/mixed-density urban scene, visually more heterogeneous
    // than flight 03's repetitive apartment-block housing. Note the
    // full downloaded dataset's satellite_coordinates_range.csv uses
    // "satelliteNN.tif" as its mapname column (map_name below), unlike
    // the example tier's bare "NN.tif" -- readUavVisLocSatelliteBounds()
    // does an exact string match, so this must match verbatim.
    samples.push_back({
        "uavvisloc_01", "uavvisloc",
        "Datasets/UAV_VisLoc_dataset/01/drone", "01", 817,
        "Datasets/UAV_VisLoc_dataset/01/01.csv",
        // ~1.19 m/px (target-mpp 1.0 default). Was briefly regenerated at
        // ~0.6 m/px to test whether map/video resolution mismatch was
        // the dominant UAV-VisLoc accuracy bottleneck -- it wasn't
        // (flat-to-worse results, see CLAUDE.md Investigation Log's
        // "Resolution-gap experiment"), so reverted back to the coarser
        // map: same accuracy, ~4x fewer pixels per reference crop, and
        // meaningfully faster full-flight runs with no downside found.
        "Images/map_clean_uavvisloc_01.png",
        "Datasets/UAV_VisLoc_dataset/satellite_ coordinates_range.csv",
        "satellite01.tif",
        "Videos/dataset_uavvisloc_01.avi",
    });

    for (auto& s : getFlight03Sample())
        samples.push_back(std::move(s));

    return samples;
}
