#!/usr/bin/env python3
"""
UAV-VisLoc altitude-datum fix: per-still height ABOVE GROUND (AGL).

UAV-VisLoc's `height` column is NOT above-ground on elevated terrain (the README gives no
datum): flights 05/06/10/11 report 773-2572 m over terrain at ~480-1950 m. The C++ dataset
pipeline seeds its GSD/crop scale from that height, so those flights were matched at a
~2.6-6.3x wrong scale (see INVESTIGATION_LOG.md, "Altitude-Datum Bug").

AGL = height - terrain(lat, lon), terrain from Copernicus GLO-30 (30 m DSM, EGM2008 heights,
public AWS bucket), bilinearly sampled under each still's GT position. That GT-position
sampling is a stand-in for an onboard laser/radar altimeter (which e.g. ALTO carries); a
GNSS-denied deployment would instead use baro - DEM(estimated position) or the altimeter
directly. Residual datum caveats: `height` may be ellipsoidal (geoid undulation in China is
tens of metres at most, a few % of AGL), and GLO-30 is a surface model (canopy/buildings).

Outputs: CSV Files/agl_uavvisloc_<flight>.csv  (Num,Height,Terrain,AGL)
         and prints per-flight mean AGL (what masks/<video>_altitude.txt should hold).
Tiles cached in Datasets/DEM/copernicus_glo30/ (gitignored with Datasets/).

Usage: python3 scripts/compute_uavvisloc_agl.py --flights 05,06,10,11 [--write-altitude-cache]
"""
import argparse
import csv
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import tifffile

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "Datasets" / "UAV_VisLoc_dataset"
DEM_DIR = REPO / "Datasets" / "DEM" / "copernicus_glo30"
CSV_DIR = REPO / "CSV Files"
MASKS = REPO / "masks"
BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_name(la, lo):
    ns, ew = ("N" if la >= 0 else "S"), ("E" if lo >= 0 else "W")
    return f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM"


_cache = {}


def load_tile(la, lo):
    if (la, lo) in _cache:
        return _cache[(la, lo)]
    name = tile_name(la, lo)
    path = DEM_DIR / f"{name}.tif"
    if not path.exists():
        DEM_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {name} ...", flush=True)
        urllib.request.urlretrieve(f"{BASE}/{name}/{name}.tif", path)
    # Pixels via OpenCV (tifffile needs the extra `imagecodecs` package for GLO-30's
    # floating-point predictor); geo-referencing read from the GeoTIFF tags, not assumed.
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float64)
    with tifffile.TiffFile(path) as tf:
        tags = tf.pages[0].tags
        sx, sy = tags["ModelPixelScaleTag"].value[:2]
        tie = tags["ModelTiepointTag"].value          # (i, j, k, X, Y, Z)
    geo = (tie[3] - tie[0] * sx, tie[4] + tie[1] * sy, sx, sy)   # lon0, lat0 of pixel (0,0) corner
    _cache[(la, lo)] = (arr, geo)
    return _cache[(la, lo)]


def terrain(lat, lon):
    """Bilinear sample at (lat, lon), pixel centres at corner + (idx + 0.5) * scale."""
    la, lo = math.floor(lat), math.floor(lon)
    a, (lon0, lat0, sx, sy) = load_tile(la, lo)
    H, W = a.shape
    r = (lat0 - lat) / sy - 0.5
    c = (lon - lon0) / sx - 0.5
    r0, c0 = int(min(max(math.floor(r), 0), H - 2)), int(min(max(math.floor(c), 0), W - 2))
    fr, fc = r - r0, c - c0
    return ((1 - fr) * (1 - fc) * a[r0, c0] + (1 - fr) * fc * a[r0, c0 + 1]
            + fr * (1 - fc) * a[r0 + 1, c0] + fr * fc * a[r0 + 1, c0 + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", default="05,06,10,11")
    ap.add_argument("--write-altitude-cache", action="store_true",
                    help="overwrite masks/Videos_dataset_uavvisloc_<f>.avi_altitude.txt with mean AGL")
    a = ap.parse_args()
    for fl in a.flights.split(","):
        rows = list(csv.DictReader(open(DATA / fl / f"{fl}.csv", newline="")))
        out = []
        for r in rows:
            h = float(r["height"])
            t = terrain(float(r["lat"]), float(r["lon"]))
            out.append((int(r["num"]), h, t, h - t))
        agl = np.array([o[3] for o in out])
        with open(CSV_DIR / f"agl_uavvisloc_{fl}.csv", "w", newline="") as fo:
            w = csv.writer(fo)
            w.writerow(["Num", "Height", "Terrain", "AGL"])
            w.writerows([(n, f"{h:.2f}", f"{t:.2f}", f"{g:.2f}") for n, h, t, g in out])
        hmean = np.mean([o[1] for o in out])
        print(f"flight {fl}: height mean {hmean:7.1f} | terrain {np.mean([o[2] for o in out]):7.1f} "
              f"(range {min(o[2] for o in out):.0f}-{max(o[2] for o in out):.0f}) | AGL mean {agl.mean():6.1f} "
              f"std {agl.std():5.1f} min {agl.min():6.1f} | scale error of raw height {hmean / agl.mean():.2f}x")
        if a.write_altitude_cache:
            p = MASKS / f"Videos_dataset_uavvisloc_{fl}.avi_altitude.txt"
            old = p.read_text().strip() if p.exists() else "(none)"
            p.write_text(f"{agl.mean():.3f}\n")
            print(f"  {p.name}: {old} -> {agl.mean():.3f}")


if __name__ == "__main__":
    main()
