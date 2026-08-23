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

    // UAV-VisLoc flight folder layout (see Datasets/UAV_VisLoc_dataset/README_dataset.txt):
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
//
// Points at the full UAV_VisLoc_dataset bundle, not the old 2GB
// UAV_VisLoc_example sample tier (that folder was deleted as part of the
// consolidation flagged in CLAUDE.md's "What Still Needs to Be Done" #9 --
// this left the config dangling until fixed here) -- map_name follows the
// full bundle's "satelliteNN.tif" convention, not the example tier's bare
// "NN.tif" (see the flight 01 comment below for why that distinction
// matters).
inline std::vector<DatasetSampleConfig> getFlight03Sample() { return {{
    "uavvisloc_03", "uavvisloc",
    "Datasets/UAV_VisLoc_dataset/03/drone", "03", 768,
    "Datasets/UAV_VisLoc_dataset/03/03.csv",
    "Images/map_clean_uavvisloc_03.png",  // ~1.19 m/px -- see flight 01's comment below re: reverting from _fine
    "Datasets/UAV_VisLoc_dataset/satellite_ coordinates_range.csv",
    "satellite03.tif",
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

    // STRATEGY.md Phase 1 ("clean reference imagery" -- see CLAUDE.md's
    // Phase 1 Investigation Log entry): same flights, same telemetry, same
    // assembled video (video_path deliberately points at the SAME cached
    // .avi as the "uavvisloc_01"/"uavvisloc_03" entries above -- video
    // assembly is independent of which satellite source is used, and
    // assembleDatasetVideo() is cached on that path already existing, so
    // this reuses it instead of re-assembling ~800 images again). Only the
    // satellite reference map differs: Esri World Imagery instead of the
    // dataset's own GeoTIFF, fetched via scripts/fetch_esri_imagery.py at
    // the IDENTICAL bounding box and pixel dimensions (so identical mpp) as
    // the baseline map it's being compared against -- imagery provenance is
    // the one deliberately changing variable.
    //
    // sample_name deliberately does NOT contain "uavvisloc_01"/"uavvisloc_03"
    // as a substring: argv[3]'s dataset-mode filter (Main.cpp) does a plain
    // substring match against sample_name (documented there -- lets
    // run_benchmark.py pass e.g. "01" to mean flight 01), so a name like
    // "uavvisloc_01_esri" would silently ALSO match the existing baseline
    // command `--flights uavvisloc_01,uavvisloc_03`, corrupting reproduction
    // of the already-recorded Phase 0 baseline. "esri01"/"esri03" avoid that.
    samples.push_back({
        "esri01", "uavvisloc",
        "Datasets/UAV_VisLoc_dataset/01/drone", "01", 817,
        "Datasets/UAV_VisLoc_dataset/01/01.csv",
        "Images/map_clean_esri_01.png",
        "Datasets/esri_coordinates_range.csv",
        "esri01",
        "Videos/dataset_uavvisloc_01.avi",
    });
    samples.push_back({
        "esri03", "uavvisloc",
        "Datasets/UAV_VisLoc_dataset/03/drone", "03", 768,
        "Datasets/UAV_VisLoc_dataset/03/03.csv",
        "Images/map_clean_esri_03.png",
        "Datasets/esri_coordinates_range.csv",
        "esri03",
        "Videos/dataset_uavvisloc_03.avi",
    });

    // STRATEGY.md Phase 2 prerequisite: the 3 validation flights promised by
    // STRATEGY.md Sec 6.1 ("3 more, choose in Phase 0") but never assigned --
    // Phase 0 explicitly deferred this. Picked from the 9 remaining UAV-VisLoc
    // flights by the same altitude-spread AGL-plausibility check that ruled
    // out flights 05/06/11 when flight 01 was originally chosen as the
    // second dev flight (large/ambiguous spread -> `height` may be MSL, not
    // AGL, over mountainous terrain -- would silently corrupt GSD/footprint
    // math): all of 02/04/07/08/09/10 have tight spreads (<10m) and are
    // AGL-plausible. Chose 04 (Taizhou-6, 738 frames) -- deliberately the
    // same broad region as dev flight 03, testing whether a fix that worked
    // there holds up on a second, independent flight nearby, not just the
    // one it was tuned against -- plus 08 (Huzhou-3, 1033 frames) and 10
    // (Huailai, 144 frames, smaller sample but a genuinely new region), both
    // real generalization tests since neither region appears in the dev set.
    // Flight 07 (only 30 frames) and 09 (same region as 08) were considered
    // and passed over -- see CLAUDE.md's Phase 2 Investigation Log for the
    // full comparison table. Held-out (sealed until Phase 4, per Sec 6.1):
    // {02, 05, 06, 07, 09, 11} -- note 05/06/11 carry the same
    // altitude-ambiguity caveat that excluded them here, so Phase 4 should
    // treat results on those three with that in mind, not as a fresh
    // problem to solve then.
    samples.push_back({
        "uavvisloc_04", "uavvisloc",
        "Datasets/UAV_VisLoc_dataset/04/drone", "04", 738,
        "Datasets/UAV_VisLoc_dataset/04/04.csv",
        "Images/map_clean_uavvisloc_04.png",
        "Datasets/UAV_VisLoc_dataset/satellite_ coordinates_range.csv",
        "satellite04.tif",
        "Videos/dataset_uavvisloc_04.avi",
    });
    samples.push_back({
        "uavvisloc_08", "uavvisloc",
        "Datasets/UAV_VisLoc_dataset/08/drone", "08", 1033,
        "Datasets/UAV_VisLoc_dataset/08/08.csv",
        "Images/map_clean_uavvisloc_08.png",
        "Datasets/UAV_VisLoc_dataset/satellite_ coordinates_range.csv",
        "satellite08.tif",
        "Videos/dataset_uavvisloc_08.avi",
    });
    samples.push_back({
        "uavvisloc_10", "uavvisloc",
        "Datasets/UAV_VisLoc_dataset/10/drone", "10", 144,
        "Datasets/UAV_VisLoc_dataset/10/10.csv",
        "Images/map_clean_uavvisloc_10.png",
        "Datasets/UAV_VisLoc_dataset/satellite_ coordinates_range.csv",
        "satellite10.tif",
        "Videos/dataset_uavvisloc_10.avi",
    });

    return samples;
}
