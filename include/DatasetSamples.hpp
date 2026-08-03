#pragma once

#include <string>
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

// Flight 03 is deliberately NOT in the active list below. Its map covers
// ~64 km^2 (vs. flight 01's ~20 km^2) -- a much bigger reference-crop
// database to search every frame -- which makes full-flight runs far slower
// than flight 01's for no diagnostic benefit once you already have a
// baseline: flight 03's numbers are fully captured in CLAUDE.md's
// Investigation Log and don't need to be re-run for comparisons that only
// need ONE flight with an existing ORB baseline (e.g. AKAZE vs ORB). Re-add
// this entry (paths still valid, nothing else needs to change) if a test
// specifically needs flight 03 again.
//
// inline std::vector<DatasetSampleConfig> getFlight03Sample() { return {{
//     "uavvisloc_03", "uavvisloc",
//     "Datasets/UAV_VisLoc_example/03/drone", "03", 768,
//     "Datasets/UAV_VisLoc_example/03/03.csv",
//     "Images/map_clean_uavvisloc_03_fine.png",
//     "Datasets/UAV_VisLoc_example/satellite_ coordinates_range.csv",
//     "03.tif",
//     "Videos/dataset_uavvisloc_03.avi",
// }}; }

inline std::vector<DatasetSampleConfig> getDatasetSamples() {
    return {
        // Flight 01 (Changjiang-20): second flight tested specifically to
        // separate "flight 03 is content-hard" from "the pipeline is
        // broken" -- see CLAUDE.md Investigation Log. Content is a river/
        // dockside/mixed-density urban scene, visually more heterogeneous
        // than flight 03's repetitive apartment-block housing. Note the
        // full downloaded dataset's satellite_coordinates_range.csv uses
        // "satelliteNN.tif" as its mapname column (map_name below), unlike
        // the example tier's bare "NN.tif" -- readUavVisLocSatelliteBounds()
        // does an exact string match, so this must match verbatim.
        {
            "uavvisloc_01", "uavvisloc",
            "Datasets/UAV_VisLoc_dataset/01/drone", "01", 817,
            "Datasets/UAV_VisLoc_dataset/01/01.csv",
            "Images/map_clean_uavvisloc_01_fine.png",  // ~0.6 m/px, see flight 03's comment above
            "Datasets/UAV_VisLoc_dataset/satellite_ coordinates_range.csv",
            "satellite01.tif",
            "Videos/dataset_uavvisloc_01.avi",
        },
    };
}
