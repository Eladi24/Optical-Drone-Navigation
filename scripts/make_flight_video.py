#!/usr/bin/env python3
"""
Stitches a UAV-VisLoc flight's ordered drone images into a normal, watchable
MP4 -- just for fun, not part of the navigation pipeline (that's
DatasetVideoAssembler, which deliberately uses a nominal fps and isn't meant
for viewing). This one plays at a constant, smooth frame rate and burns in
a small telemetry overlay (frame number, elapsed real time, altitude) read
from the flight's own metadata CSV.

Usage:
    python3 scripts/make_flight_video.py \
        --image-dir "Datasets/UAV_VisLoc_dataset/03/drone" \
        --prefix 03 --frame-count 768 \
        --telemetry-csv "Datasets/UAV_VisLoc_dataset/03/03.csv" \
        --out "Videos/flight_03_fun.mp4" --fps 24 --width 1280
"""
import argparse
import csv

import cv2


def load_telemetry(csv_path):
    rows = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows[int(r["num"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--prefix", required=True, help="Flight prefix, e.g. '03' for 03_0001.JPG")
    ap.add_argument("--frame-count", type=int, required=True)
    ap.add_argument("--telemetry-csv", help="Optional flight metadata CSV for the on-screen overlay")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=1280, help="Output width; height keeps aspect ratio")
    args = ap.parse_args()

    telemetry = load_telemetry(args.telemetry_csv) if args.telemetry_csv else {}
    t0 = None
    writer = None

    for i in range(1, args.frame_count + 1):
        path = f"{args.image_dir}/{args.prefix}_{i:04d}.JPG"
        frame = cv2.imread(path)
        if frame is None:
            print(f"skipping missing frame {path}")
            continue

        h, w = frame.shape[:2]
        out_h = round(args.width * h / w)
        frame = cv2.resize(frame, (args.width, out_h))

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out, fourcc, args.fps, (args.width, out_h))

        row = telemetry.get(i)
        if row:
            overlay = f"frame {i}/{args.frame_count}   alt {float(row['height']):.0f}m"
            cv2.putText(frame, overlay, (20, out_h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, overlay, (20, out_h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 1, cv2.LINE_AA)

        writer.write(frame)

    if writer is not None:
        writer.release()
        print(f"Wrote {args.out} ({args.frame_count} frames @ {args.fps}fps, "
              f"~{args.frame_count / args.fps:.1f}s)")
    else:
        print("No frames written -- check --image-dir/--prefix/--frame-count.")


if __name__ == "__main__":
    main()
