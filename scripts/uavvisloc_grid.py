#!/usr/bin/env python3
"""
Shared UAV-VisLoc satellite-map bounds/grid/crop helpers, factored out so
scripts/prepare_finetune_dataset.py and scripts/eval_retrieval_recall.py don't
duplicate the same grid math. Mirrors generateReferenceCropsGrid()
(src/VideoProcessing.cpp:15-91, params from src/Main.cpp:637-671) closely
enough to be accurate to well under a metre, confirmed this session by an
Explore pass over the real C++ source -- using the simpler bounds-direct
formulation noted there (min/max_lat = RB_lat_map/LT_lat_map directly, no mpp
roundtrip) rather than re-deriving bounds from mpp*rows/cols.

GRID_SPACING_M / CROP_SIZE_M match the live C++ pipeline's own UAV-VisLoc
constants (200m spacing, 300m crop footprint) -- not independently chosen
here.
"""
import csv
import math

import numpy as np

M_PER_DEG_LAT = 111320.0
GRID_SPACING_M = 200.0
CROP_SIZE_M = 300.0

COORDS_CSV = "Datasets/UAV_VisLoc_dataset/satellite_ coordinates_range.csv"


def m_per_deg_lng(lat_deg: float) -> float:
    return M_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def load_bounds(mapname: str, coords_csv: str = COORDS_CSV):
    """Returns (lt_lat, lt_lon, rb_lat, rb_lon) for `mapname` (e.g. "satellite02.tif"),
    mirroring readUavVisLocSatelliteBounds() (src/TelemetryImporter.cpp:154-171)."""
    with open(coords_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["mapname"] == mapname:
                return (float(row["LT_lat_map"]), float(row["LT_lon_map"]),
                        float(row["RB_lat_map"]), float(row["RB_lon_map"]))
    raise KeyError(f"{mapname!r} not found in {coords_csv}")


def generate_grid(lt_lat, lt_lon, rb_lat, rb_lon,
                   grid_spacing_m: float = GRID_SPACING_M):
    """Returns (lats, lngs) arrays of grid-cell centres covering the bounds,
    same spacing/origin convention as generateReferenceCropsGrid(). Does not
    replicate the C++ code's >50%-in-raster-bounds crop-clipping check (a
    Python-side crop() call below simply clips at the PNG's own edges,
    matching the same edge behaviour np.clip would give -- confirmed a minor,
    edge-only effect, not relevant for interior grid cells)."""
    min_lat, max_lat = rb_lat, lt_lat
    min_lng, max_lng = lt_lon, rb_lon
    center_lat = (lt_lat + rb_lat) / 2.0
    mlng = m_per_deg_lng(center_lat)
    spacing_lat = grid_spacing_m / M_PER_DEG_LAT
    spacing_lng = grid_spacing_m / mlng

    lat_vals = np.arange(min_lat, max_lat, spacing_lat)
    lng_vals = np.arange(min_lng, max_lng, spacing_lng)
    lats, lngs = np.meshgrid(lat_vals, lng_vals, indexing="ij")
    return lats.ravel(), lngs.ravel()


def latlng_to_px(lat, lng, lt_lat, lt_lon, rb_lat, rb_lon, width_px, height_px):
    px = (lng - lt_lon) / (rb_lon - lt_lon) * width_px
    py = (lat - lt_lat) / (rb_lat - lt_lat) * height_px
    return px, py


def crop_at(map_img, lat, lng, bounds, crop_size_m: float = CROP_SIZE_M):
    """Crops `map_img` (a loaded cv2 BGR array) at (lat, lng), `crop_size_m` on
    a side. `bounds` is the (lt_lat, lt_lon, rb_lat, rb_lon) tuple from
    load_bounds(). Returns None if the crop would be entirely outside the
    raster. mpp is derived per-axis from the raster's own real dimensions
    (not assumed isotropic), matching how the C++ pipeline's already-corrected
    (equirectangular-anisotropy-fixed) PNGs are square-pixel in practice."""
    lt_lat, lt_lon, rb_lat, rb_lon = bounds
    h, w = map_img.shape[:2]
    mpp_x = ((rb_lon - lt_lon) * m_per_deg_lng((lt_lat + rb_lat) / 2.0)) / w
    mpp_y = ((lt_lat - rb_lat) * M_PER_DEG_LAT) / h
    half_px_x = int(round((crop_size_m / mpp_x) / 2.0))
    half_px_y = int(round((crop_size_m / mpp_y) / 2.0))

    px, py = latlng_to_px(lat, lng, lt_lat, lt_lon, rb_lat, rb_lon, w, h)
    px, py = int(round(px)), int(round(py))
    x0, x1 = max(0, px - half_px_x), min(w, px + half_px_x)
    y0, y1 = max(0, py - half_px_y), min(h, py + half_px_y)
    if x1 <= x0 or y1 <= y0:
        return None
    return map_img[y0:y1, x0:x1]


def nearest_grid_index(lat, lng, grid_lats, grid_lngs):
    """Index into (grid_lats, grid_lngs) of the cell nearest (lat, lng), by
    plain squared lat/lng distance (fine at this scale -- cells are ~200m
    apart, no need for a real haversine here)."""
    d2 = (grid_lats - lat) ** 2 + (grid_lngs - lng) ** 2
    return int(np.argmin(d2))
