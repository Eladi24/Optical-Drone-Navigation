#!/usr/bin/env python3
"""
Converts a UAV-VisLoc satellite GeoTIFF (e.g. satellite03.tif, ~2.5GB / ~850
megapixels at ~0.3m/px) into a downsampled PNG that cv::imread can load
directly, for use as processVideoNavigation()'s reference_map.

The raw GeoTIFFs are far too large to decode in one shot on a typical dev
machine (a full in-memory RGB decode is ~2.5GB+ and there is no guarantee
that much free RAM is available) and are stored as uncompressed *tiled*
TIFFs, which OpenCV/PIL don't stream. This script reads the file lazily via
tifffile's zarr view (only the tiles actually touched are decoded, one
horizontal band of tiles at a time) and box-downsamples each band before it
is written to the output array, so peak memory stays at "one band", not
"whole image".

The dataset's satellite_coordinates_range.csv is left as the single source
of truth for the map's real-world bounding box -- this script does not write
a sidecar; the C++ ingestion code re-derives ground-sample-distance from
that CSV plus the output PNG's own pixel dimensions.

IMPORTANT -- equirectangular anisotropy correction: the raw GeoTIFF is a
simple lat/lon-linear (equirectangular) raster with square pixels in DEGREE
space (confirmed via its embedded ModelPixelScaleTag: ~2.6822637e-06 deg/px
in both axes), not in real-world METRES. At this dataset's ~32.3 deg
latitude, one degree of longitude covers cos(32.3deg) =~ 84.5% of the ground
distance one degree of latitude does, so naively downsampling by the same
integer factor in both axes produces a PNG whose horizontal and vertical
metres/pixel differ by ~18%. The rest of this pipeline (generateReferenceCropsGrid,
processVideoNavigation's coordinate math, all inherited from the Haifa/Google-Maps
pipeline) assumes a single isotropic mpp -- true for Google's Mercator-projected
static map tiles, false for this raw equirectangular TIFF. Left uncorrected, this
silently shifts every computed pixel location by hundreds of metres over a
multi-km map (verified empirically: ~380m at this dataset's frame-1 location).
Fixed here by resizing the initial (isotropic-factor) downsample so its final
mpp_x equals mpp_y, using the latitude-invariant mpp_y (metres/degree-latitude
is a true constant, 111320) as the reference scale.

Usage:
    python3 scripts/convert_uavvisloc_satellite.py \
        --tif "Datasets/UAV_VisLoc_example/03/satellite03.tif" \
        --coords-csv "Datasets/UAV_VisLoc_example/satellite_ coordinates_range.csv" \
        --mapname satellite03.tif \
        --out "Images/map_clean_uavvisloc_03.png"
"""
import argparse
import csv
import math
import sys

import numpy as np
import tifffile
import zarr
from PIL import Image

M_PER_DEG_LAT = 111320.0


def m_per_deg_lng(lat_deg: float) -> float:
    return M_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def native_mpp(coords_csv: str, mapname: str, width_px: int, height_px: int):
    with open(coords_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["mapname"].strip() == mapname:
                lt_lat, lt_lon = float(row["LT_lat_map"]), float(row["LT_lon_map"])
                rb_lat, rb_lon = float(row["RB_lat_map"]), float(row["RB_lon_map"])
                avg_lat = (lt_lat + rb_lat) / 2.0
                width_m = (rb_lon - lt_lon) * m_per_deg_lng(avg_lat)
                height_m = (lt_lat - rb_lat) * M_PER_DEG_LAT
                return width_m / width_px, height_m / height_px
    raise SystemExit(f"'{mapname}' not found in {coords_csv}")


def downsample_tiled_tiff(tif_path: str, factor: int, band_tile_rows: int = 4) -> np.ndarray:
    za = tifffile.imread(tif_path, aszarr=True)
    z = zarr.open(za, mode="r")
    h, w, c = z.shape
    w_trim = (w // factor) * factor
    h_trim = (h // factor) * factor
    out = np.empty((h_trim // factor, w_trim // factor, c), dtype=np.uint8)

    band_rows = 256 * band_tile_rows
    for y0 in range(0, h_trim, band_rows):
        y1 = min(y0 + band_rows, h_trim)
        bh = (y1 - y0) // factor * factor
        y1c = y0 + bh
        band = z[y0:y1c, :w_trim, :]
        reduced = band.reshape(bh // factor, factor, w_trim // factor, factor, c).mean(axis=(1, 3)).astype(np.uint8)
        out[y0 // factor: y0 // factor + reduced.shape[0], :, :] = reduced
        print(f"  band {y0}:{y1c} / {h_trim}", file=sys.stderr, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", required=True, help="Path to the raw satelliteNN.tif")
    ap.add_argument("--coords-csv", required=True, help="Path to satellite_coordinates_range.csv")
    ap.add_argument("--mapname", required=True, help="mapname value to look up, e.g. satellite03.tif")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--target-mpp", type=float, default=1.0,
                     help="Desired output resolution in metres/pixel (default 1.0, matching the "
                          "Haifa pipeline's typical reference-map resolution)")
    args = ap.parse_args()

    with tifffile.TiffFile(args.tif) as tf:
        h, w = tf.pages[0].shape[:2]

    mpp_x, mpp_y = native_mpp(args.coords_csv, args.mapname, w, h)
    native_mpp_avg = (mpp_x + mpp_y) / 2.0
    factor = max(1, round(args.target_mpp / native_mpp_avg))
    print(f"Native resolution: ~{native_mpp_avg:.3f} m/px ({w}x{h}px). "
          f"Downsampling by {factor}x -> ~{native_mpp_avg * factor:.3f} m/px.", file=sys.stderr)

    out = downsample_tiled_tiff(args.tif, factor)

    # Correct equirectangular anisotropy (see module docstring): the initial
    # downsample has mpp_x != mpp_y since it applied the same integer factor
    # to both axes of a degree-square (not metre-square) native grid. Resize
    # width only so mpp_x matches mpp_y, using mpp_y (metres/degree-latitude
    # is a true constant) as the reference scale.
    out_mpp_x = mpp_x * factor
    out_mpp_y = mpp_y * factor
    new_width = round(out.shape[1] * out_mpp_x / out_mpp_y)
    if new_width != out.shape[1]:
        print(f"Correcting equirectangular anisotropy: mpp_x={out_mpp_x:.4f} vs "
              f"mpp_y={out_mpp_y:.4f} -> resizing width {out.shape[1]} -> {new_width}px",
              file=sys.stderr)
        out = np.array(Image.fromarray(out).resize((new_width, out.shape[0]), Image.LANCZOS))

    Image.fromarray(out).save(args.out)
    print(f"Wrote {args.out} ({out.shape[1]}x{out.shape[0]}px, "
          f"mpp~{out_mpp_y:.3f} isotropic)", file=sys.stderr)


if __name__ == "__main__":
    main()
