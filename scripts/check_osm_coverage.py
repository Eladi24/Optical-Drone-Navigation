#!/usr/bin/env python3
"""
Route #3 (RESULTS.md §8) step 1: does OpenStreetMap have enough vector structure
(buildings, roads, water) over the UAV-VisLoc flight areas to build a 2.5D /
structural reference map? UAV-VisLoc is mixed rural / peri-urban Chinese terrain,
where OSM coverage is not a given. This gates the whole route.

Queries an Overpass mirror for element counts per flight bbox (no geometry
download, just counts). Read the table: flights with hundreds+ buildings and a
real road network are viable for OSM-based structural matching; near-empty
flights would need a different structural source (e.g. a footprint-segmentation
model run on the satellite raster).

Note: overpass-api.de is unreachable from this environment; maps.mail.ru's mirror
works. Run with `python3 -u`.
"""
import csv
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).parent.parent
COORDS = REPO / "Datasets/UAV_VisLoc_dataset/satellite_ coordinates_range.csv"
ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


def query_counts(bbox):
    """bbox = (S, W, N, E) -> [buildings, roads, water] counts (or 'ERR')."""
    b = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    q = (f'[out:json][timeout:50];'
         f'way["building"]({b});out count;'
         f'way["highway"]({b});out count;'
         f'nwr["natural"="water"]({b});out count;'
         f'way["waterway"]({b});out count;')
    data = urllib.parse.urlencode({"data": q}).encode()
    for ep in ENDPOINTS:
        try:
            with urllib.request.urlopen(urllib.request.Request(ep, data=data), timeout=60) as r:
                j = json.load(r)
            return [int(e["tags"]["total"]) for e in j["elements"] if e.get("type") == "count"]
        except Exception:
            continue
    return None


def main():
    rows = list(csv.DictReader(open(COORDS, newline="")))
    print(f"{'flight':<10}{'region':<16}{'km_w':>6}{'km_h':>6}   {'bldgs':>7}{'roads':>7}{'water':>7}{'wway':>6}   verdict")
    print("-" * 82)
    for row in rows:
        name = row["mapname"].replace("satellite", "").replace(".tif", "")
        lt_lat, lt_lon = float(row["LT_lat_map"]), float(row["LT_lon_map"])
        rb_lat, rb_lon = float(row["RB_lat_map"]), float(row["RB_lon_map"])
        bbox = (rb_lat, lt_lon, lt_lat, rb_lon)
        km_h = abs(lt_lat - rb_lat) * 111.32
        km_w = abs(rb_lon - lt_lon) * 111.32 * math.cos(math.radians((lt_lat + rb_lat) / 2))
        c = query_counts(bbox)
        if c is None:
            print(f"{name:<10}{row['region']:<16}{km_w:>6.1f}{km_h:>6.1f}   {'ERR — all endpoints failed'}")
            continue
        bldg, road, water, wway = (c + [0, 0, 0, 0])[:4]
        area = max(km_w * km_h, 0.1)
        dens = bldg / area
        verdict = "GOOD" if dens > 50 else "THIN" if dens > 5 else "SPARSE"
        print(f"{name:<10}{row['region']:<16}{km_w:>6.1f}{km_h:>6.1f}   {bldg:>7}{road:>7}{water:>7}{wway:>6}   "
              f"{verdict} ({dens:.0f} bldg/km²)")
        time.sleep(4)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
