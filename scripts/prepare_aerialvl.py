#!/usr/bin/env python3
"""
AerialVL -> UAV-VisLoc-layout adapter for the zero-tuning transfer test (2026-11-23).

The frozen pipeline (git tag uavvisloc-v1) is run UNCHANGED on AerialVL, so instead of teaching every C++/Python
stage a second dataset, the two extracted AerialVL sequences are presented in UAV-VisLoc's on-disk layout:

    Datasets/AerialVL/visloc_format/<seq>/<seq>.csv            num,filename,date,lat,lon,height,Omega,Kappa,Phi1,Phi2
    Datasets/AerialVL/visloc_format/<seq>/drone/<seq>_%04d.JPG -> symlinks to the original PNGs
    Datasets/UAV_VisLoc_dataset/<seq>                          -> symlink to the above (frozen scripts use that root)
    Images/map_clean_aerialvl_large.png                         the large geo-referenced map (0.959 m/px, square)
    Datasets/aerialvl_coordinates_range.csv                     its bounds, satellite_coordinates_range.csv schema

Sequences: av1 = long_trajtr/2023-03-16-18-04-01 (639 frames, partly dusk), av2 = short_trajtr/2023-03-18-16-55-37
(498 frames). Mapping choices (documented, not tuned):
  * GT lat/lon/time come from the AerialVL filenames  @<UTC ms>@<lon>@<lat>@.png  (NovAtel GNSS, 1.5 m RMS).
  * Frames have image-top = EAST (2026-11-17 geo-ref check: rotate 90 deg CW for north-up), i.e. heading 90 deg,
    so Phi1 = Phi2 = 90 and the pipeline's default-on -Phi1 pre-warp makes them north-up.
  * No attitude data on disk (IMU is on Baidu): Omega = Kappa = 0.
  * The .JPG symlink names are cosmetic: OpenCV/PIL decode by content (they are PNGs).
  * height / AGL: AerialVL publishes neither altitude nor intrinsics here, so `calibrate-gsd` measures the frame
    GSD from GPS step length vs. the frozen stills-VO pixel displacement (fitted on av2 only, checked on av1) and
    writes an EQUIVALENT AGL = GSD * W / (2 tan(51.5/2)), so the frozen HFOV constant reproduces that GSD
    everywhere downstream. A stand-in for the altimeter + intrinsics a real platform has.

    python3 -I scripts/prepare_aerialvl.py layout
    python3 -I scripts/stills_visual_odometry.py --flights av2,av1        (frozen, unchanged)
    python3 -I scripts/prepare_aerialvl.py calibrate-gsd
"""
import argparse
import csv
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "Datasets" / "AerialVL" / "extracted"
OUT = REPO / "Datasets" / "AerialVL" / "visloc_format"
VISLOC_ROOT = REPO / "Datasets" / "UAV_VisLoc_dataset"
CSV_DIR = REPO / "CSV Files"
SEQS = {"av1": "long_trajtr/2023-03-16-18-04-01", "av2": "short_trajtr/2023-03-18-16-55-37",
        # 2026-11-24 fresh test (parts 019-021, never seen by any design decision): midday on another day, and late
        # afternoon on av1's day.
        "av3": "short_trajtr/2023-03-11-11-48-35", "av4": "short_trajtr/2023-03-16-16-58-43"}
HFOV_DEG = 51.5           # CameraModel::kUavVisLocHfovDeg -- the frozen constant
M_LAT = 111320.0
HEADER = ["num", "filename", "date", "lat", "lon", "height", "Omega", "Kappa", "Phi1", "Phi2"]


def parse_frame_name(name):
    """'@1678961172907@120.44002333333331@36.59765166666666@.png' -> (ms, lon, lat)."""
    parts = name.split("@")
    if len(parts) != 5 or not parts[4].endswith(".png"):
        raise ValueError(f"unexpected AerialVL frame name: {name}")
    return int(parts[1]), float(parts[2]), float(parts[3])


def frames_of(seq):
    d = SRC / SEQS[seq]
    rows = sorted((parse_frame_name(p.name) + (p,) for p in d.glob("@*@.png")), key=lambda r: r[0])
    return rows


def write_seq_csv(seq, rows, height):
    path = OUT / seq / f"{seq}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i, (ms, lon, lat, _) in enumerate(rows, start=1):
            date = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            w.writerow([i, f"{seq}_{i:04d}.JPG", date, f"{lat:.10f}", f"{lon:.10f}", f"{height:.3f}",
                        0, 0, 90, 90])
    return path


def cmd_layout(_args):
    OUT.mkdir(parents=True, exist_ok=True)
    for seq in SEQS:
        rows = frames_of(seq)
        dts = np.diff([r[0] for r in rows]) / 1000.0
        kx = M_LAT * math.cos(math.radians(rows[0][2]))
        steps = [math.hypot((b[1] - a[1]) * kx, (b[2] - a[2]) * M_LAT) for a, b in zip(rows, rows[1:])]
        drone = OUT / seq / "drone"
        drone.mkdir(parents=True, exist_ok=True)
        for old in drone.glob(f"{seq}_*.JPG"):
            old.unlink()
        for i, (_, _, _, png) in enumerate(rows, start=1):
            os.symlink(png.resolve(), drone / f"{seq}_{i:04d}.JPG")
        csv_path = write_seq_csv(seq, rows, height=0.0)  # height filled in by calibrate-gsd
        link = VISLOC_ROOT / seq
        if link.is_symlink():
            link.unlink()
        os.symlink(os.path.relpath(OUT / seq, VISLOC_ROOT), link)
        print(f"{seq} ({SEQS[seq]}): {len(rows)} frames -> {csv_path.relative_to(REPO)}")
        print(f"   dt median {np.median(dts):.3f} s (min {dts.min():.3f}, max {dts.max():.3f}); "
              f"GPS step median {np.median(steps):.2f} m -> {np.median(steps) / np.median(dts):.1f} m/s; "
              f"max step {max(steps):.1f} m")

    # Map: the large geo-referenced map, bounds from its filename (@large_map@LT_lon@LT_lat@RB_lon@RB_lat@.tif).
    import cv2
    import tifffile
    tif = next((SRC / "geo_referenced_map").glob("@large_map@*.tif"))
    _, _, lt_lon, lt_lat, rb_lon, rb_lat, _ = tif.name.split("@")
    img = tifffile.imread(tif)
    print(f"map {tif.name[:40]}...: shape {img.shape}, dtype {img.dtype}")
    if img.ndim == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_RGB2BGR)
    out_png = REPO / "Images" / "map_clean_aerialvl_large.png"
    cv2.imwrite(str(out_png), img)
    lat_c = (float(lt_lat) + float(rb_lat)) / 2
    w_m = (float(rb_lon) - float(lt_lon)) * M_LAT * math.cos(math.radians(lat_c))
    h_m = (float(lt_lat) - float(rb_lat)) * M_LAT
    print(f"   -> {out_png.relative_to(REPO)}: mpp x {w_m / img.shape[1]:.4f}, y {h_m / img.shape[0]:.4f} m/px")
    coords = REPO / "Datasets" / "aerialvl_coordinates_range.csv"
    with open(coords, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mapname", "LT_lat_map", "LT_lon_map", "RB_lat_map", "RB_lon_map", "region"])
        w.writerow(["aerialvl_large", lt_lat, lt_lon, rb_lat, rb_lon, "Qingdao"])
    print(f"   -> {coords.relative_to(REPO)}")


def cmd_calibrate_gsd(args):
    def pairs(seq):
        """GSD samples over non-overlapping windows of --baseline consecutive VO pairs: |GPS displacement| /
        |summed VO displacement|. A single 0.5 s step (~3.4 m) is too short: GPS/VO noise biased the
        per-pair ratio ~15% LOW on av2 (k=1 0.109 vs k=10..40 0.1305 m/px, stable), measured 2026-11-23."""
        pos = {int(r["num"]): (float(r["lat"]), float(r["lon"]))
               for r in csv.DictReader(open(OUT / seq / f"{seq}.csv", newline=""))}
        vo = {}
        for r in csv.DictReader(open(CSV_DIR / f"stills_vo_uavvisloc_{seq}.csv", newline="")):
            if r["From_Num"].startswith("#") or r["Ok"] != "1" or int(r["Inliers"]) < args.min_inliers:
                continue
            vo[int(r["From_Num"])] = np.array([float(r["Disp_X_Px"]), float(r["Disp_Y_Px"])])
        k, out = args.baseline, []
        for a in range(1, max(pos) - k + 1, k):
            if not all(i in vo for i in range(a, a + k)):
                continue
            disp_px = np.linalg.norm(sum(vo[i] for i in range(a, a + k)))
            (la, lo), (lb, lob) = pos[a], pos[a + k]
            step_m = math.hypot((lob - lo) * M_LAT * math.cos(math.radians(la)), (lb - la) * M_LAT)
            if step_m >= args.min_step_m and disp_px > 0:
                out.append((a, step_m / disp_px))
        return out

    def summarise(seq, pr):
        g = np.array([x for _, x in pr])
        q = np.array_split(g, 4)
        print(f"{seq}: {len(g)} windows of {args.baseline} pairs, GSD median {np.median(g):.5f} m/px (IQR {np.percentile(g, 25):.5f}-"
              f"{np.percentile(g, 75):.5f}); per-quarter medians " + ", ".join(f"{np.median(x):.5f}" for x in q))
        return float(np.median(g))

    fit = summarise(args.fit_seq, pairs(args.fit_seq))
    gsd = {args.fit_seq: fit}
    src = {args.fit_seq: args.fit_seq}
    for seq in SEQS:
        if seq == args.fit_seq:
            continue
        own = summarise(seq, pairs(seq))
        rel = own / fit - 1
        verdict = f"within 5% -> use the {args.fit_seq} calibration" if abs(rel) <= 0.05 else "DIFFERS > 5% -> own calibration"
        print(f"   {seq} vs {args.fit_seq}: {rel * 100:+.1f}% ({verdict})")
        gsd[seq] = fit if abs(rel) <= 0.05 else own
        src[seq] = args.fit_seq if abs(rel) <= 0.05 else f"{seq} itself (differs {rel * 100:+.1f}% from {args.fit_seq})"

    with open(OUT / "gsd_calibration.txt", "w") as f:
        for seq, g in gsd.items():
            width = 2048
            agl = g * width / (2 * math.tan(math.radians(HFOV_DEG / 2)))
            f.write(f"{seq} gsd_m_per_px={g:.6f} agl_equiv_m={agl:.3f} (calibrated on {src[seq]})\n")
            rows = frames_of(seq)
            write_seq_csv(seq, rows, height=agl)
            with open(CSV_DIR / f"agl_uavvisloc_{seq}.csv", "w", newline="") as af:
                w = csv.writer(af)
                w.writerow(["Num", "Height", "Terrain", "AGL"])
                for i in range(1, len(rows) + 1):
                    w.writerow([i, f"{agl:.3f}", 0, f"{agl:.3f}"])
            print(f"{seq}: GSD {g:.5f} m/px -> equivalent AGL {agl:.1f} m (HFOV {HFOV_DEG}); footprint "
                  f"{g * width:.0f} x {g * 1536:.0f} m; wrote CSV height + agl_uavvisloc_{seq}.csv")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("layout")
    c = sub.add_parser("calibrate-gsd")
    c.add_argument("--fit-seq", default="av2")
    c.add_argument("--min-inliers", type=int, default=30)
    c.add_argument("--min-step-m", type=float, default=20.0)
    c.add_argument("--baseline", type=int, default=20, help="VO pairs per GSD sample (see pairs())")
    args = ap.parse_args()
    {"layout": cmd_layout, "calibrate-gsd": cmd_calibrate_gsd}[args.cmd](args)


if __name__ == "__main__":
    main()
