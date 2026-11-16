#!/usr/bin/env python3
"""
RESULTS.md route #1 bridge experiment, step 1: frame-to-frame visual odometry from
UAV-VisLoc's consecutive survey stills.

Why this exists: the fusion thesis (odometry + retrieval anchors + robust MAP fusion)
has never been tested on REAL retrieval anchors, because no dataset on disk has both
continuous VIO-grade odometry and wide-area retrieval. UAV-VisLoc's consecutive stills
are ~80m apart against a ~570m footprint (~86% overlap), and registering them is a
SAME-domain problem (drone-vs-drone, seconds apart), which the self-referential
experiment showed is easy. This is NOT a stand-in for VIO drift characteristics (the
earlier objection still holds -- see INVESTIGATION_LOG.md, Phase 4 Step 3); it is a
camera-only relative-motion source that lets us ask the decisive question: given good
odometry, are our real anchors good enough to pin the trajectory to tens of metres?

Per consecutive pair (i -> i+1): SIFT on a 1/4-res decode, Lowe ratio, 4-DOF
(similarity) RANSAC via cv2.estimateAffinePartial2D. Outputs, in FULL-RES pixels and
the IMAGE frame of still i (x right, y down): the ground displacement of the nadir
point (centre of i+1 expressed in i's pixels, minus centre of i), plus the relative
rotation/scale and inlier count. No GPS / telemetry is read here -- conversion to
metres (barometric altitude + HFOV) and to the world frame (compass heading) happens
downstream in stills_odometry_fusion.py, where the calibration is fitted on dev
flights only.

Output: CSV Files/stills_vo_uavvisloc_<flight>.csv
Usage:  python3 scripts/stills_visual_odometry.py [--flights 01,03,...] [--workers N]
"""
import argparse
import csv
import math
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "Datasets" / "UAV_VisLoc_dataset"
OUT = REPO / "CSV Files"
ALL_FLIGHTS = ["01", "02", "03", "04", "05", "06", "08", "10", "11"]

N_FEATURES = 3000
RATIO = 0.8
RANSAC_PX = 3.0          # at 1/4 resolution
MIN_INLIERS = 30         # below this the pair is reported as failed (ok=0)


def load_reduced(path):
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return img


def full_res_width(path):
    # One full decode per flight, to convert 1/4-res pixels back to full-res pixels exactly.
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return im.shape[1], im.shape[0]


def register(kp1, des1, kp2, des2, matcher, shape):
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None
    knn = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < 10:
        return None
    p1 = np.float32([kp1[m.queryIdx].pt for m in good])
    p2 = np.float32([kp2[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC,
                                         ransacReprojThreshold=RANSAC_PX,
                                         maxIters=5000, confidence=0.999)
    if M is None:
        return None
    n_inl = int(inl.sum())
    A, t = M[:, :2], M[:, 2]
    h, w = shape
    c = np.array([w / 2.0, h / 2.0])
    # Point in image i that lands on the centre of image i+1.
    if abs(np.linalg.det(A)) < 1e-6:   # degenerate RANSAC model -> treat as failed pair
        return None
    p = np.linalg.solve(A, c - t)
    disp = p - c
    rot = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    scale = math.hypot(A[0, 0], A[1, 0])
    return len(good), n_inl, disp, rot, scale


def run_flight(flight):
    d = DATA / flight
    rows = list(csv.DictReader(open(d / f"{flight}.csv", newline="")))
    files = [d / "drone" / r["filename"] for r in rows]
    nums = [int(r["num"]) for r in rows]
    full_w, full_h = full_res_width(files[0])

    sift = cv2.SIFT_create(nfeatures=N_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    out_rows = []
    prev = None
    for k, f in enumerate(files):
        img = load_reduced(f)
        kp, des = sift.detectAndCompute(img, None)
        cur = (kp, des, img.shape)
        if prev is not None:
            s = full_w / prev[2][1]
            res = register(prev[0], prev[1], kp, des, matcher, prev[2])
            if res is None:
                out_rows.append([nums[k - 1], nums[k], 0, 0, "", "", "", "", 0])
            else:
                n_good, n_inl, disp, rot, scale = res
                ok = int(n_inl >= MIN_INLIERS)
                out_rows.append([nums[k - 1], nums[k], n_good, n_inl,
                                 f"{disp[0] * s:.3f}", f"{disp[1] * s:.3f}",
                                 f"{rot:.4f}", f"{scale:.5f}", ok])
        prev = cur
        if k % 100 == 0:
            print(f"[{flight}] {k}/{len(files)}", flush=True)

    out_path = OUT / f"stills_vo_uavvisloc_{flight}.csv"
    with open(out_path, "w", newline="") as fo:
        wr = csv.writer(fo)
        wr.writerow(["From_Num", "To_Num", "Matches", "Inliers", "Disp_X_Px", "Disp_Y_Px",
                     "Rot_Deg", "Scale", "Ok"])
        wr.writerow(["#full_res", full_w, full_h, "", "", "", "", "", ""])
        wr.writerows(out_rows)
    n_ok = sum(r[-1] for r in out_rows)
    print(f"[{flight}] done: {n_ok}/{len(out_rows)} pairs ok -> {out_path.name}", flush=True)
    return flight, n_ok, len(out_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", default=",".join(ALL_FLIGHTS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cv2.setNumThreads(1)
    with Pool(a.workers) as pool:
        for fl, ok, n in pool.imap_unordered(run_flight, a.flights.split(",")):
            print(f"== flight {fl}: {ok}/{n} ok")


if __name__ == "__main__":
    main()
