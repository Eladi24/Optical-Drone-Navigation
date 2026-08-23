#!/usr/bin/env python3
"""
Phase 3 data prep: extracts MUN-FRL's quarry1 ROS bag (Datasets/MUN_FRL_quarry1/
flight_dataset2.bag) into two consumable forms:

  1. EuRoC MAV format (Datasets/MUN_FRL_quarry1/mav0/) -- the de facto standard
     input layout most existing open-source VIO implementations (VINS-Fusion,
     OpenVINS, Kimera-VIO, basalt, ...) already support natively. STRATEGY.md's
     Phase 3 task is explicit: "do not write a VIO from scratch, integrate an
     existing implementation" -- this format is what makes that integration a
     later drop-in rather than a bespoke-parser problem.
       mav0/cam0/data/<timestamp_ns>.png   -- from /camera/image_color only
                                               (/camera/image_mono is the same
                                               physical feed, redundant, skipped)
       mav0/cam0/data.csv                  -- timestamp_ns,filename
       mav0/imu0/data.csv                  -- timestamp_ns,w_x,w_y,w_z,a_x,a_y,a_z
       mav0/leica0/data.csv                -- timestamp_ns,p_x,p_y,p_z (local ENU
                                               metres). EuRoC's recognized
                                               POSITION-ONLY ground-truth substream
                                               (used when full 6-DoF state incl.
                                               orientation/velocity/bias isn't
                                               available) -- a deliberate format
                                               adaptation, not a literal full EuRoC
                                               state_groundtruth_estimate0 stream,
                                               since the PPK source only has
                                               position.

  2. This project's own established schema (see TelemetryImporter.hpp for the
     exact convention this mirrors):
       CSV Files/ground_truth_munfrl_quarry1.csv  (Frame,Time_Sec,Lat,Lng,Label)
       CSV Files/imu_munfrl_quarry1.csv           (Time_Sec,ax,ay,az,gx,gy,gz) --
         a new schema (this project has never had dense inertial data before),
         kept separate from telemetry_<sample>.csv's existing per-frame
         Alt/Heading schema rather than overloading it.

TIME ALIGNMENT (the real problem this script exists to solve correctly, not
assume away): every topic's header.stamp in this bag decodes to 2018-01-28,
but the flight's own PPK ground truth and onboard DJI flight log both
independently agree the real flight was 2022-02-22 -- the recording computer's
system clock was evidently unsynced at record time (common un-NTP'd
embedded-Linux symptom), consistently across every topic (one shared, wrongly
set clock, not a per-sensor problem). Naive epoch-based alignment between the
bag and the PPK file would therefore be silently wrong by ~4 years. Fixed by
solving for a constant offset Delta (bag_time + Delta = PPK's time reference)
that best aligns the bag's own live /fix GPS stream against the PPK track --
both trace the same real trajectory, just on different clocks, so position
agreement (not any assumption about the "true" epoch) pins down Delta. This
also absorbs, for free, any GPST-vs-UTC labeling difference between the two
GPS sources (PPK's header says GPST; NMEA-derived /fix is conventionally UTC;
~18s apart as of this flight's date) -- deliberately not reasoned about
separately, since the offset search is robust to any constant bias.

Usage:
    source ../CV_IP/cv_env/bin/activate
    python3 scripts/extract_munfrl_bag.py
"""
import csv
import sys
from calendar import timegm
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_ground_truth import M_PER_DEG_LAT, m_per_deg_lng

REPO = Path(__file__).parent.parent
BAG_PATH = REPO / "Datasets" / "MUN_FRL_quarry1" / "flight_dataset2.bag"
PPK_PATH = REPO / "Datasets" / "MUN_FRL_quarry1" / "ppk_data" / "quarry_1_ppk.pos"
MAV_DIR = REPO / "Datasets" / "MUN_FRL_quarry1" / "mav0"
CSV_DIR = REPO / "CSV Files"
SAMPLE_NAME = "munfrl_quarry1"

CAM_TOPIC = "/camera/image_color"
IMU_TOPIC = "/imu/data"
FIX_TOPIC = "/fix"


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec / 1e9


def project_xy(lat, lng, lat0, lng0):
    mlng = m_per_deg_lng(lat0)
    x = (lng - lng0) * mlng
    y = (lat - lat0) * M_PER_DEG_LAT
    return x, y


def load_ppk(path):
    """RTKLIB .pos file: '%'-prefixed header/comment lines, then
    'YYYY/MM/DD HH:MM:SS.sss lat lon height Q ns ...' data rows (space-separated).
    Q=1 is a fixed RTK solution (cm-level); kept as-is here, not filtered, since
    quarry1's whole track was Q=1 per the earlier manual inspection."""
    t, lat, lng, height = [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("%") or not line.strip():
                continue
            parts = line.split()
            dt = datetime.strptime(parts[0] + " " + parts[1], "%Y/%m/%d %H:%M:%S.%f")
            t.append(timegm(dt.timetuple()) + dt.microsecond / 1e6)
            lat.append(float(parts[2])); lng.append(float(parts[3])); height.append(float(parts[4]))
    order = np.argsort(t)
    return (np.array(t)[order], np.array(lat)[order], np.array(lng)[order], np.array(height)[order])


def load_fix(reader, conns):
    t, lat, lng, alt = [], [], [], []
    for _, _, rawdata in reader.messages(connections=[conns[FIX_TOPIC]]):
        msg = reader.deserialize(rawdata, conns[FIX_TOPIC].msgtype)
        t.append(stamp_to_sec(msg.header.stamp))
        lat.append(msg.latitude); lng.append(msg.longitude); alt.append(msg.altitude)
    return np.array(t), np.array(lat), np.array(lng), np.array(alt)


def solve_time_offset(fix_t, fix_lat, fix_lng, ppk_t, ppk_lat, ppk_lng):
    """Finds Delta such that fix_t + Delta aligns with ppk_t, by minimizing
    summed squared position residual (in a shared local-ENU frame) between the
    /fix track and the PPK track linearly interpolated at the shifted times.
    Coarse-to-fine grid search: the initial guess (matching each track's first
    sample) gets the ~4-year wrong-epoch gap; the refinement nails the
    sub-second alignment."""
    lat0, lng0 = ppk_lat.mean(), ppk_lng.mean()
    fx, fy = project_xy(fix_lat, fix_lng, lat0, lng0)
    px, py = project_xy(ppk_lat, ppk_lng, lat0, lng0)

    def residual(delta):
        shifted = fix_t + delta
        valid = (shifted >= ppk_t[0]) & (shifted <= ppk_t[-1])
        if valid.sum() < len(fix_t) * 0.5:
            return np.inf, 0
        ix = np.interp(shifted[valid], ppk_t, px)
        iy = np.interp(shifted[valid], ppk_t, py)
        res = np.hypot(fx[valid] - ix, fy[valid] - iy)
        return float(np.sqrt(np.mean(res ** 2))), int(valid.sum())

    delta0 = ppk_t[0] - fix_t[0]
    best_delta, best_rmse = delta0, np.inf
    for lo, hi, step in [(delta0 - 30, delta0 + 30, 0.5), (0, 0, 0.02)]:
        if lo == hi:  # refine pass, centered on current best
            lo, hi = best_delta - 1.0, best_delta + 1.0
        for cand in np.arange(lo, hi + step, step):
            rmse, n = residual(cand)
            if rmse < best_rmse:
                best_rmse, best_delta = rmse, cand
    final_rmse, final_n = residual(best_delta)
    return best_delta, final_rmse, final_n


def main():
    print(f"Loading PPK ground truth from {PPK_PATH} ...")
    ppk_t, ppk_lat, ppk_lng, ppk_h = load_ppk(PPK_PATH)
    print(f"  {len(ppk_t)} PPK rows, {ppk_t[-1]-ppk_t[0]:.1f}s span")

    with AnyReader([BAG_PATH]) as reader:
        conns = {c.topic: c for c in reader.connections}

        print(f"Loading /fix from bag ...")
        fix_t, fix_lat, fix_lng, fix_alt = load_fix(reader, conns)
        print(f"  {len(fix_t)} /fix rows, {fix_t[-1]-fix_t[0]:.1f}s span (bag clock)")

        print("Solving time offset (bag clock -> PPK time reference) ...")
        delta, rmse, n_matched = solve_time_offset(fix_t, fix_lat, fix_lng, ppk_t, ppk_lat, ppk_lng)
        print(f"  Delta = {delta:.3f}s  (~{delta/86400:.1f} days)")
        print(f"  post-alignment RMSE = {rmse:.2f}m over {n_matched}/{len(fix_t)} matched /fix samples")
        if rmse > 50.0:
            print("  WARNING: residual is much larger than expected for two independent GPS "
                  "solutions of the same antenna -- do not trust the alignment before investigating.")

        (MAV_DIR / "cam0" / "data").mkdir(parents=True, exist_ok=True)
        (MAV_DIR / "imu0").mkdir(parents=True, exist_ok=True)
        (MAV_DIR / "leica0").mkdir(parents=True, exist_ok=True)
        CSV_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Extracting {CAM_TOPIC} frames ...")
        cam_rows = []  # (corrected_sec, filename)
        for i, (_, _, rawdata) in enumerate(reader.messages(connections=[conns[CAM_TOPIC]])):
            msg = reader.deserialize(rawdata, conns[CAM_TOPIC].msgtype)
            corrected_sec = stamp_to_sec(msg.header.stamp) + delta
            ts_ns = int(round(corrected_sec * 1e9))
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            fname = f"{ts_ns}.png"
            cv2.imwrite(str(MAV_DIR / "cam0" / "data" / fname), img)
            cam_rows.append((corrected_sec, ts_ns, fname))
            if (i + 1) % 500 == 0:
                print(f"  {i+1} frames...")
        print(f"  {len(cam_rows)} frames written to {MAV_DIR / 'cam0' / 'data'}")

        print(f"Extracting {IMU_TOPIC} ...")
        imu_rows = []  # (corrected_sec, ts_ns, wx,wy,wz, ax,ay,az)
        for _, _, rawdata in reader.messages(connections=[conns[IMU_TOPIC]]):
            msg = reader.deserialize(rawdata, conns[IMU_TOPIC].msgtype)
            corrected_sec = stamp_to_sec(msg.header.stamp) + delta
            ts_ns = int(round(corrected_sec * 1e9))
            w, a = msg.angular_velocity, msg.linear_acceleration
            imu_rows.append((corrected_sec, ts_ns, w.x, w.y, w.z, a.x, a.y, a.z))
        print(f"  {len(imu_rows)} IMU rows")

    cam_rows.sort()
    imu_rows.sort()

    with open(MAV_DIR / "cam0" / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#timestamp [ns]", "filename"])
        for _, ts_ns, fname in cam_rows:
            w.writerow([ts_ns, fname])

    with open(MAV_DIR / "imu0" / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]",
                    "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"])
        for _, ts_ns, wx, wy, wz, ax, ay, az in imu_rows:
            w.writerow([ts_ns, wx, wy, wz, ax, ay, az])

    lat0, lng0 = ppk_lat.mean(), ppk_lng.mean()
    ppk_x, ppk_y = project_xy(ppk_lat, ppk_lng, lat0, lng0)
    with open(MAV_DIR / "leica0" / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#timestamp [ns]", "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"])
        for ts, x, y, h in zip(ppk_t, ppk_x, ppk_y, ppk_h):
            w.writerow([int(round(ts * 1e9)), x, y, h])

    # Project-native schema (see TelemetryImporter.hpp for the exact convention
    # mirrored here): Time_Sec is real elapsed seconds, rebased to 0 at the
    # earliest corrected timestamp across camera+IMU, shared by both files.
    #
    # PPK's covered window (~172s) is narrower than the camera's full span
    # (~231s) -- frames outside [ppk_t[0], ppk_t[-1]] have no real ground truth,
    # and np.interp would silently CLAMP them to the nearest PPK endpoint rather
    # than erroring, which would write a flat, wrong position into the ground
    # truth file. Skip those frames explicitly instead (same "don't silently
    # extrapolate ground truth" discipline evaluate_ground_truth.py's own
    # max_frame_gap skip already applies).
    t0 = min(cam_rows[0][0], imu_rows[0][0])
    in_range = [(i, c) for i, c in enumerate(cam_rows) if ppk_t[0] <= c[0] <= ppk_t[-1]]
    skipped_frames = len(cam_rows) - len(in_range)
    in_range_t = [c[0] for _, c in in_range]
    ppk_lat_interp = np.interp(in_range_t, ppk_t, ppk_lat)
    ppk_lng_interp = np.interp(in_range_t, ppk_t, ppk_lng)

    with open(CSV_DIR / f"ground_truth_{SAMPLE_NAME}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Frame", "Time_Sec", "Lat", "Lng", "Label"])
        for (frame_i, (corrected_sec, _, _)), lat, lng in zip(in_range, ppk_lat_interp, ppk_lng_interp):
            w.writerow([frame_i + 1, f"{corrected_sec - t0:.3f}", f"{lat:.9f}", f"{lng:.9f}", ""])

    with open(CSV_DIR / f"imu_{SAMPLE_NAME}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time_Sec", "ax", "ay", "az", "gx", "gy", "gz"])
        for corrected_sec, _, wx, wy, wz, ax, ay, az in imu_rows:
            w.writerow([f"{corrected_sec - t0:.6f}", ax, ay, az, wx, wy, wz])

    print("\n=== Validation summary ===")
    print(f"Frames: {len(cam_rows)} (bag had 4627 /camera/image_color messages)")
    print(f"IMU rows: {len(imu_rows)} (bag had 92542 /imu/data messages)")
    print(f"Time offset Delta: {delta:.3f}s, alignment RMSE: {rmse:.2f}m ({n_matched} samples)")
    cam_span = cam_rows[-1][0] - cam_rows[0][0]
    ppk_span = ppk_t[-1] - ppk_t[0]
    print(f"Corrected camera timestamp range: [{cam_rows[0][0]:.1f}, {cam_rows[-1][0]:.1f}] "
          f"({cam_span:.1f}s), PPK range: [{ppk_t[0]:.1f}, {ppk_t[-1]:.1f}] ({ppk_span:.1f}s)")
    print(f"Ground-truth CSV: {len(in_range)}/{len(cam_rows)} frames have real PPK coverage "
          f"({skipped_frames} frames outside the PPK window were skipped, not extrapolated)")
    print(f"EuRoC export: {MAV_DIR}")
    print(f"Project-native export: {CSV_DIR / f'ground_truth_{SAMPLE_NAME}.csv'}, "
          f"{CSV_DIR / f'imu_{SAMPLE_NAME}.csv'}")


if __name__ == "__main__":
    main()
