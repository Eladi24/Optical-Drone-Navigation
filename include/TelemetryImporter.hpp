#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Converts a UAV-VisLoc flight's own metadata CSV into the pipeline's
// existing CSV schemas, so the rest of the pipeline can consume it exactly
// like Haifa's hand-annotated ground truth, with no downstream changes.
//
// Input format (UAV-VisLoc's <flight>/<flight>.csv, see its README_example.txt):
//   num,filename,date,lat,lon,height,Omega,Kappa,Phi1,Phi2
//   Omega=pitch, Kappa=roll, Phi1/Phi2=yaw (Phi1 has higher confidence).
//
// Output:
//   "CSV Files/ground_truth_<sample_name>.csv"  (Frame,Time_Sec,Lat,Lng,Label)
//     -- same schema Haifa's manual annotator produces, so readGroundTruthStart()
//        and scripts/evaluate_ground_truth.py work completely unchanged. Frame is
//        the CSV's own 1-based "num" column, matching the frame index the
//        DatasetVideoAssembler's synthetic video uses for the same image.
//   "CSV Files/telemetry_<sample_name>.csv"     (Frame,Time_Sec,Lat,Lng,Alt,Heading)
//     -- denser per-frame companion (every row, not just a "Label"-worthy subset;
//        UAV-VisLoc already has one row per image so in practice this has the
//        same row count as the ground-truth file, but keeps the two concerns --
//        "a verified point in the existing schema" vs. "camera-model input" --
//        separate). Heading is Phi1, exported for future use but not yet
//        consumed anywhere (not wired into the Kalman filter in this pass --
//        its sign/reference-frame convention relative to CoordinateUtils'
//        heading hasn't been verified, and getting that wrong would silently
//        corrupt filtering in a way indistinguishable from a real accuracy
//        problem, exactly the ambiguity this dataset pivot exists to remove).
//
// Time_Sec in both files is the row's capture timestamp minus the first row's,
// so it starts at 0 like a video's own frame_idx/fps clock -- and is real,
// not assumed-constant, which is what processVideoNavigation()'s optional
// frame_times_sec parameter is for (capture intervals in this dataset vary,
// e.g. survey-flight turnarounds take much longer between frames).
// ---------------------------------------------------------------------------
struct TelemetryImportResult {
    bool   success;
    int    frame_count;
    double mean_altitude_m;   // mean of the height column; <=0 if unavailable
};

TelemetryImportResult importUavVisLocTelemetry(const std::string& csv_path,
                                                const std::string& sample_name);

// Reads back "CSV Files/telemetry_<sample_name>.csv" (written by
// importUavVisLocTelemetry above) into a vector indexed by (Frame - 1), i.e.
// by the 0-based frame index the assembled video's frame_idx uses. Pass the
// result to processVideoNavigation()'s frame_times_sec parameter. Returns an
// empty vector if the file doesn't exist or has no rows.
std::vector<double> loadTelemetryFrameTimes(const std::string& sample_name);

// Looks up a satellite map's real-world bounding box from UAV-VisLoc's own
// satellite_coordinates_range.csv (format: mapname,LT_lat_map,LT_lon_map,
// RB_lat_map,RB_lon_map,region -- see README_example.txt).
struct SatelliteBounds {
    bool   success;
    double lt_lat, lt_lon, rb_lat, rb_lon;
};

SatelliteBounds readUavVisLocSatelliteBounds(const std::string& coords_csv_path,
                                             const std::string& map_name);
