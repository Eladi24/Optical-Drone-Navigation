#!/usr/bin/env python3
"""
Fetches an Esri World Imagery crop covering the SAME real-world bounding box
as an existing UAV-VisLoc satellite map (as produced by
convert_uavvisloc_satellite.py), for STRATEGY.md Phase 1 ("clean reference
imagery" -- see CLAUDE.md's Phase 1 investigation entry for why: testing
whether an independent, non-Google-Maps imagery provider changes matching
accuracy, holding geography and resolution fixed as the one variable that
changes).

Output pixel dimensions are copied directly from --reference-png, so the
fetched image has IDENTICAL mpp_x/mpp_y (including the equirectangular
anisotropy correction already baked into that reference image's aspect
ratio) to the map it's meant to be compared against -- no separate mpp math
needed here, and no risk of accidentally introducing a resolution-mismatch
confound (STRATEGY.md's "Resolution-gap experiment" already showed that
matters).

Esri's export endpoint caps image dimensions per call (4096x4096 as of this
writing, confirmed via the service's own f=json metadata) -- both UAV-VisLoc
flight maps exceed that in at least one axis, so this tiles the request into
--max-tile-sized chunks and stitches them, the same structural idea as
convert_uavvisloc_satellite.py's own band-tiled GeoTIFF read, just tiling
over HTTP calls instead of GeoTIFF rows.

Writes a bounds row into --out-coords-csv using the IDENTICAL 6-column
schema UAV-VisLoc's own satellite_coordinates_range.csv uses
(mapname,LT_lat_map,LT_lon_map,RB_lat_map,RB_lon_map,region), so
readUavVisLocSatelliteBounds() (src/TelemetryImporter.cpp) reads it
completely unmodified -- the C++ ingestion side needs zero changes.

Usage:
    python3 scripts/fetch_esri_imagery.py \
        --coords-csv "Datasets/UAV_VisLoc_dataset/satellite_ coordinates_range.csv" \
        --mapname satellite01.tif \
        --reference-png "Images/map_clean_uavvisloc_01.png" \
        --out "Images/map_clean_esri_01.png" \
        --out-coords-csv "Datasets/esri_coordinates_range.csv" \
        --out-mapname esri01
"""
import argparse
import csv
import io
import math
import os
import sys
import time

import requests
from PIL import Image

DEFAULT_SERVICE_URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"


def read_bounds(coords_csv: str, mapname: str):
    with open(coords_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["mapname"].strip() == mapname:
                return {
                    "lt_lat": float(row["LT_lat_map"]),
                    "lt_lon": float(row["LT_lon_map"]),
                    "rb_lat": float(row["RB_lat_map"]),
                    "rb_lon": float(row["RB_lon_map"]),
                    "region": row.get("region", "").strip(),
                }
    raise SystemExit(f"'{mapname}' not found in {coords_csv}")


def tile_ranges(total: int, max_tile: int):
    """Contiguous [start, end) pixel ranges covering [0, total), each <= max_tile."""
    n = math.ceil(total / max_tile)
    edges = [round(i * total / n) for i in range(n + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n)]


def fetch_tile(service_url: str, bbox, size_px, timeout=60):
    x0, y0, x1, y1 = bbox
    params = {
        "bbox": f"{x0},{y0},{x1},{y1}",
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{size_px[0]},{size_px[1]}",
        "format": "png24",
        "f": "image",
    }
    resp = requests.get(service_url, params=params, timeout=timeout)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    if img.size != tuple(size_px):
        raise RuntimeError(f"Esri returned {img.size}, requested {size_px}")
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords-csv", required=True, help="Source bounds CSV (e.g. UAV-VisLoc's satellite_coordinates_range.csv)")
    ap.add_argument("--mapname", required=True, help="mapname value to look up in --coords-csv, e.g. satellite01.tif")
    ap.add_argument("--reference-png", required=True, help="Existing baseline map PNG whose pixel dimensions (and therefore mpp) this fetch should match")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--out-coords-csv", required=True, help="Bounds CSV to create/append a row to, same schema as --coords-csv")
    ap.add_argument("--out-mapname", required=True, help="mapname key to write for the new row, e.g. esri01")
    ap.add_argument("--service-url", default=DEFAULT_SERVICE_URL, help="ArcGIS REST export endpoint")
    ap.add_argument("--max-tile", type=int, default=4000, help="Max width/height per HTTP request (service cap is 4096)")
    ap.add_argument("--sleep", type=float, default=0.5, help="Delay between tile requests (politeness on a free public service)")
    args = ap.parse_args()

    bounds = read_bounds(args.coords_csv, args.mapname)
    with Image.open(args.reference_png) as ref:
        target_width, target_height = ref.size
    print(f"Target size (from {args.reference_png}): {target_width}x{target_height}px", file=sys.stderr)

    lt_lat, lt_lon = bounds["lt_lat"], bounds["lt_lon"]
    rb_lat, rb_lon = bounds["rb_lat"], bounds["rb_lon"]

    # Fetch at the bbox's NATIVE (uncorrected) degree aspect ratio, not
    # target_width/target_height directly. ArcGIS's cached MapServer export
    # operation silently ADJUSTS the requested bbox to match the requested
    # image size's aspect ratio when the two don't already agree, rather than
    # stretching pixels to fit -- confirmed empirically (a seam-crossing crop
    # comparison showed large, unambiguous duplicate content -- identical
    # moving-boat positions -- straddling a tile boundary, while the
    # existing baseline map at the same coordinates showed genuinely
    # different content there). target_width/height are deliberately
    # anisotropy-corrected (see convert_uavvisloc_satellite.py's docstring)
    # and so do NOT match the bbox's raw lon/lat degree-span aspect ratio,
    # which triggers exactly that server-side adjustment and made adjacent
    # tiles silently cover overlapping ground. Fixed the same way
    # convert_uavvisloc_satellite.py handles the identical anisotropy
    # problem: fetch at the aspect ratio the request actually matches, then
    # do the anisotropy correction as one final width-only resize.
    height_px_native = target_height
    lon_span = rb_lon - lt_lon
    lat_span = lt_lat - rb_lat
    width_px_native = max(1, round(height_px_native * lon_span / lat_span))
    print(f"Native (aspect-matched) fetch size: {width_px_native}x{height_px_native}px "
          f"(bbox aspect {lon_span/lat_span:.4f})", file=sys.stderr)

    canvas = Image.new("RGB", (width_px_native, height_px_native))
    x_ranges = tile_ranges(width_px_native, args.max_tile)
    y_ranges = tile_ranges(height_px_native, args.max_tile)
    total_tiles = len(x_ranges) * len(y_ranges)
    done = 0

    for (py0, py1) in y_ranges:
        for (px0, px1) in x_ranges:
            # Linear interpolation in degree space: lon increases with x, lat
            # decreases with y (LT = top-left = north-west corner) -- same
            # convention convert_uavvisloc_satellite.py's output already
            # uses, so tile bboxes here are consistent with it. Each tile's
            # own request aspect ratio matches its own sub-bbox's aspect
            # ratio too, since it's a proportional slice of an
            # already-aspect-correct canvas -- keeps every individual HTTP
            # call free of the same server-side bbox-adjustment problem.
            lon0 = lt_lon + lon_span * (px0 / width_px_native)
            lon1 = lt_lon + lon_span * (px1 / width_px_native)
            lat0 = lt_lat - lat_span * (py0 / height_px_native)  # north edge of this tile
            lat1 = lt_lat - lat_span * (py1 / height_px_native)  # south edge of this tile

            tile_w, tile_h = px1 - px0, py1 - py0
            print(f"Fetching tile {done + 1}/{total_tiles}: "
                  f"px=({px0},{py0})-({px1},{py1}) bbox=({lon0:.6f},{lat1:.6f},{lon1:.6f},{lat0:.6f})",
                  file=sys.stderr)
            tile_img = fetch_tile(args.service_url, (lon0, lat1, lon1, lat0), (tile_w, tile_h))
            canvas.paste(tile_img, (px0, py0))
            done += 1
            if done < total_tiles:
                time.sleep(args.sleep)

    if width_px_native != target_width:
        print(f"Correcting equirectangular anisotropy: resizing width "
              f"{width_px_native} -> {target_width}px (height unchanged)", file=sys.stderr)
        canvas = canvas.resize((target_width, target_height), Image.LANCZOS)

    canvas.save(args.out)
    print(f"Wrote {args.out} ({canvas.width}x{canvas.height}px)", file=sys.stderr)

    write_header = not os.path.exists(args.out_coords_csv)
    with open(args.out_coords_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["mapname", "LT_lat_map", "LT_lon_map", "RB_lat_map", "RB_lon_map", "region"])
        w.writerow([args.out_mapname, lt_lat, lt_lon, rb_lat, rb_lon, bounds["region"]])
    print(f"Appended bounds row '{args.out_mapname}' to {args.out_coords_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
