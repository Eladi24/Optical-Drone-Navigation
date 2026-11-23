#!/usr/bin/env python3
"""
Camera-tilt correction for coarse-to-fine (c2f) positions -- physics, no fitting (2026-11-23).

c2f localises the ground point at the IMAGE CENTRE (proto_coarse_to_fine.py: "camera nadir, tilt ignored").
If the GPS/GT is the drone itself, a tilted camera puts the image centre AGL * tan(tilt) away from the drone:
    image_centre = drone + AGL * (tan(a_fwd) * forward + tan(a_right) * right)
with forward = the image-up direction = compass heading Phi1 (verified by the -Phi1 north-up pre-warp) and
right = forward rotated 90 deg clockwise. The UAV-VisLoc README says Omega = pitch, Kappa = roll, but not the axis
or sign conventions, so all 8 combinations (which angle is along-track x two signs) are scored, ONE is chosen
on flight 01 only, and it is applied unchanged to every other flight. Diagnostic: the least-squares gain of the
residual on the predicted offset should be ~1.0 if the model (and "GT = drone position") is right.

    python3 -I scripts/c2f_tilt_correction.py --json <step2.json> --select 01
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "Datasets" / "UAV_VisLoc_dataset"
CSV_DIR = REPO / "CSV Files"
M_LAT = 111320.0


def load_telemetry(flight):
    """filename-index (0-based) -> dict(lat, lon, agl, omega, kappa, phi1). AGL as proto_coarse_to_fine uses it."""
    agl = {}
    ap = CSV_DIR / f"agl_uavvisloc_{flight}.csv"
    if ap.exists():
        agl = {int(r["Num"]): float(r["AGL"]) for r in csv.DictReader(open(ap, newline=""))}
    out = {}
    for r in csv.DictReader(open(DATA / flight / f"{flight}.csv", newline="")):
        idx = int(r["filename"].rsplit("_", 1)[1].split(".")[0]) - 1
        out[idx] = dict(lat=float(r["lat"]), lon=float(r["lon"]),
                        agl=agl.get(int(r["num"]), float(r["height"])),
                        omega=float(r["Omega"]), kappa=float(r["Kappa"]), phi1=float(r["Phi1"]))
    return out


def predicted_offset_en(t, swap, s_fwd, s_right):
    """Image-centre minus drone, metres (E, N), for one convention."""
    a_fwd, a_right = (t["kappa"], t["omega"]) if swap else (t["omega"], t["kappa"])
    d_fwd = s_fwd * t["agl"] * math.tan(math.radians(a_fwd))
    d_right = s_right * t["agl"] * math.tan(math.radians(a_right))
    h = math.radians(t["phi1"])
    fwd = (math.sin(h), math.cos(h))          # (E, N)
    right = (math.cos(h), -math.sin(h))
    return (d_fwd * fwd[0] + d_right * right[0], d_fwd * fwd[1] + d_right * right[1])


def residuals(rows, tel):
    """c2f estimate minus GT, metres (E, N), for accepted stills."""
    out = []
    for r in rows:
        t = tel[r["idx"]]
        e = (r["est_lon"] - t["lon"]) * M_LAT * math.cos(math.radians(t["lat"]))
        n = (r["est_lat"] - t["lat"]) * M_LAT
        out.append((r, t, np.array([e, n])))
    return out


def score(res, conv):
    errs, pred, resid = [], [], []
    for _, t, v in res:
        p = np.array(predicted_offset_en(t, *conv))
        errs.append(np.linalg.norm(v - p))
        pred.append(p)
        resid.append(v)
    pred, resid = np.array(pred), np.array(resid)
    gain = float((pred * resid).sum() / max((pred * pred).sum(), 1e-9))   # LS gain of residual on prediction
    return np.array(errs), gain


def conv_name(conv):
    swap, sf, sr = conv
    a_fwd, a_right = ("Kappa", "Omega") if swap else ("Omega", "Kappa")
    return f"fwd={'+' if sf > 0 else '-'}{a_fwd} right={'+' if sr > 0 else '-'}{a_right}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="proto_coarse_to_fine.py --out JSON (Step 2, --prior fused)")
    ap.add_argument("--select", default="01", help="flight used to CHOOSE the convention (dev)")
    args = ap.parse_args()

    d = json.load(open(args.json))
    flights = sorted({r["flight"] for r in d["results"]})
    by_flight = {f: [r for r in d["results"] if r["flight"] == f] for f in flights}
    tel = {f: load_telemetry(f) for f in flights}
    convs = list(itertools.product([False, True], [1, -1], [1, -1]))

    sel = residuals([r for r in by_flight[args.select] if r["accepted"]], tel[args.select])
    base = np.median([np.linalg.norm(v) for *_, v in sel])
    print(f"Convention search on flight {args.select} only (accepted stills n={len(sel)}, uncorrected median {base:.1f} m):")
    table = []
    for c in convs:
        errs, gain = score(sel, c)
        table.append((np.median(errs), c, gain))
    for med, c, gain in sorted(table, key=lambda x: x[0]):
        print(f"   {conv_name(c):28s} median {med:5.1f} m   gain {gain:+.2f}")
    chosen = min(table, key=lambda x: x[0])[1]
    print(f"CHOSEN (frozen for all flights): {conv_name(chosen)}\n")

    for f in flights:
        res = residuals([r for r in by_flight[f] if r["accepted"]], tel[f])
        before = np.array([np.linalg.norm(v) for *_, v in res])
        after, gain = score(res, chosen)
        best_here = min(convs, key=lambda c: np.median(score(res, c)[0]))
        tilt_sd = np.std([math.hypot(t["omega"], t["kappa"]) for _, t, _ in res])
        tag = "(selection flight)" if f == args.select else "(no fitting)"
        print(f"flight {f} {tag}: accepted n={len(res)}, tilt magnitude sd {tilt_sd:.1f} deg, gain {gain:+.2f}")
        print(f"   accepted-still err  mean/med/p90  {before.mean():5.1f}/{np.median(before):5.1f}/{np.percentile(before, 90):5.1f} m"
              f"  ->  {after.mean():5.1f}/{np.median(after):5.1f}/{np.percentile(after, 90):5.1f} m"
              f"   A@10 {np.mean(before <= 10) * 100:.0f}% -> {np.mean(after <= 10) * 100:.0f}%")
        # Deliverable over ALL stills: corrected c2f where accepted, else the fused prior.
        corr = {r["idx"]: e for (r, _, _), e in zip(res, after)}
        allb = np.array([r["err"] if r["accepted"] else r["prior_err"] for r in by_flight[f]])
        alla = np.array([corr[r["idx"]] if r["accepted"] else r["prior_err"] for r in by_flight[f]])
        prior = np.array([r["prior_err"] for r in by_flight[f]])
        print(f"   ALL stills mean/med/p90: prior {prior.mean():.0f}/{np.median(prior):.0f}/{np.percentile(prior, 90):.0f}"
              f" -> c2f {allb.mean():.0f}/{np.median(allb):.0f}/{np.percentile(allb, 90):.0f}"
              f" -> c2f+tilt {alla.mean():.0f}/{np.median(alla):.0f}/{np.percentile(alla, 90):.0f} m")
        print(f"   (best convention on this flight alone, for consistency only: {conv_name(best_here)})")


if __name__ == "__main__":
    main()
