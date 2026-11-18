#!/usr/bin/env python3
"""
Coarse-to-fine registration, Step 1 (RESULTS.md route #1 follow-up, 2026-11-17).

Question: given a rough position prior (~150-200 m, what stills-VO + local consensus delivers),
can each still be registered to the satellite map accurately enough to reach TENS of metres?
Today's per-frame anchors carry ~100-160 m error even when the chosen crop is correct: weak
8-DOF homographies from ~13-15 inliers on a 1.2 m/px map with 256 keypoints (diagnosed
2026-11-17 -- no sign/axis bug; projecting through H is worse than taking the crop centre).

This step isolates REGISTRATION from prior quality: the prior is GPS + random offset
(per-axis sigma = SIGMA/sqrt(2), i.e. ~SIGMA m RMS radial). Step 2 swaps in the real fused prior.

Per still:
  1. Known geometry removes most unknowns: scale from AGL + measured 51.5 deg HFOV, rotation
     from the compass (rotate by -Phi1 with cv2.getRotationMatrix2D -- measured; the C++
     pre-warp's +heading is the opposite sign). The still is resampled north-up at the map's working resolution.
  2. Map keypoints are extracted ONCE per (flight, resolution) on a fixed 384-px tile grid
     (DISK's ONNX graph is fixed at 384x384 / <=256 keypoints; tiling gives more keypoints and
     finer resolution without re-exporting). Only map keypoints inside prior +- WINDOW_M are used.
  3. DISK+LightGlue (same ONNX models as the C++ DiskLightGlueMatching), then a 4-DOF
     similarity (cv2.estimateAffinePartial2D, RANSAC). Accepted if inliers >= MIN_INLIERS and
     the fitted scale/rotation stay near identity (they should: both are already removed).
  4. Position = the still's centre (camera nadir, tilt ignored for now) pushed through the fit.

Map sources (one variable at a time):
  --map png      Images/map_clean_uavvisloc_<f>.png (~1.2 m/px, what the pipeline uses)
  --map geotiff  Datasets/UAV_VisLoc_dataset/<f>/satellite<f>.tif (0.30 m/px native, read
                 lazily via zarr, resampled to --res); only 01/03/09 still have it on disk.
Both sources are linear in lat/lon across pixels (the PNG is the GeoTIFF downsampled), so the
same window code serves both.

Usage (dev flights only -- held-out is a later, one-time gate):
  python3 -I scripts/proto_coarse_to_fine.py --flights 01,03 --map geotiff --res 0.6
  python3 -I scripts/proto_coarse_to_fine.py --flights 01,03 --map png
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import tifffile
import zarr

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "Datasets" / "UAV_VisLoc_dataset"
COORDS = DATA / "satellite_ coordinates_range.csv"
CSV_DIR = REPO / "CSV Files"
M_LAT = 111320.0
HFOV_DEG = 51.5          # CameraModel::kUavVisLocHfovDeg (measured 2026-11-16)
TILE = 384               # DISK ONNX input size
SIGMA_M = 150.0          # prior error, RMS radial
WINDOW_M = 300.0         # search radius around the prior (2 sigma)
MIN_INLIERS = 8           # set 2026-11-17 from 20 flight-01 stills: good fits had 10-48 inliers, failures <= 5
RANSAC_PX = 4.0
MAX_SCALE_DEV = 0.15     # |scale - 1|
MAX_ROT_DEG = 25.0       # residual rotation after compass north-up; good fits show -5..-18 deg
                         # (compass deviation + crab angle), so 15 deg rejected correct fits


def m_lng(lat):
    return M_LAT * math.cos(math.radians(lat))


# ----------------------------------------------------------------------------- models
class DiskLightGlue:
    def __init__(self, threads):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        p = ["CPUExecutionProvider"]
        self.disk = ort.InferenceSession(str(REPO / "models" / "disk.onnx"), so, providers=p)
        self.lg = ort.InferenceSession(str(REPO / "models" / "disk_lightglue.onnx"), so, providers=p)

    def tile_features(self, rgb_tile):
        """rgb_tile: HxWx3 uint8 RGB, H,W <= TILE (zero-padded here). Returns kpts (n,2) float
        in tile pixels, scores (n,), desc (n,128)."""
        h, w = rgb_tile.shape[:2]
        pad = np.zeros((TILE, TILE, 3), np.uint8)
        pad[:h, :w] = rgb_tile
        x = (pad.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        k, s, d = self.disk.run(None, {"image": x})
        k, s, d = k[0].astype(np.float32), s[0], d[0]
        keep = (k[:, 0] < w - 2) & (k[:, 1] < h - 2) & (k[:, 0] > 1) & (k[:, 1] > 1)
        return k[keep], s[keep], d[keep]

    def features(self, rgb, valid=None):
        """Tile an arbitrary-size RGB image; drop keypoints on invalid (e.g. rotation-border) pixels."""
        H, W = rgb.shape[:2]
        K, S, D = [], [], []
        for y0 in range(0, H, TILE):
            for x0 in range(0, W, TILE):
                t = rgb[y0:y0 + TILE, x0:x0 + TILE]
                if valid is not None and valid[y0:y0 + TILE, x0:x0 + TILE].mean() < 0.2:
                    continue
                k, s, d = self.tile_features(t)
                k = k + np.array([x0, y0], np.float32)
                K.append(k); S.append(s); D.append(d)
        if not K:
            return np.zeros((0, 2), np.float32), np.zeros(0, np.float32), np.zeros((0, 128), np.float32)
        K, S, D = np.concatenate(K), np.concatenate(S), np.concatenate(D)
        if valid is not None:
            xi = np.clip(K[:, 0].astype(int), 0, W - 1)
            yi = np.clip(K[:, 1].astype(int), 0, H - 1)
            m = valid[yi, xi] > 0
            # also require a margin: DISK responds to the hard rotation border
            er = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8))
            m &= er[yi, xi] > 0
            K, S, D = K[m], S[m], D[m]
        return K, S, D

    def match(self, k0, d0, size0, k1, d1, size1):
        """LightGlue on whole-image keypoints. Normalisation as LightGlue's own
        normalize_keypoints: (k - size/2) / (max(size)/2)."""
        def norm(k, size):
            w, h = size
            return ((k - np.array([w / 2, h / 2], np.float32)) / (max(w, h) / 2)).astype(np.float32)
        m0, _, ms0, _ = self.lg.run(None, {"kpts0": norm(k0, size0)[None], "kpts1": norm(k1, size1)[None],
                                           "desc0": d0[None].astype(np.float32),
                                           "desc1": d1[None].astype(np.float32)})
        m0 = m0[0]
        i = np.where(m0 >= 0)[0]
        return i, m0[i], ms0[0][i]


# ----------------------------------------------------------------------------- map
class MapSource:
    """Pixel grid linear in lat/lon. Global keypoints are extracted on a TILE grid at `res`
    metres/pixel and cached; `res_px` maps (row, col) of the resampled grid to lat/lon."""

    def __init__(self, flight, kind, res):
        with open(COORDS, newline="") as f:
            for r in csv.DictReader(f):
                if r["mapname"] == f"satellite{flight}.tif":
                    self.lt_lat, self.lt_lon = float(r["LT_lat_map"]), float(r["LT_lon_map"])
                    self.rb_lat, self.rb_lon = float(r["RB_lat_map"]), float(r["RB_lon_map"])
        self.kind, self.flight = kind, flight
        if kind == "png":
            img = cv2.imread(str(REPO / "Images" / f"map_clean_uavvisloc_{flight}.png"))
            self.src = img[:, :, ::-1]          # RGB
            self.src_bgr = True
        else:
            tif = tifffile.TiffFile(str(DATA / flight / f"satellite{flight}.tif"))
            self.src = zarr.open(tif.aszarr(), mode="r")
            if isinstance(self.src, zarr.hierarchy.Group):
                self.src = self.src["0"]
        self.SH, self.SW = self.src.shape[:2]
        lat_c = (self.lt_lat + self.rb_lat) / 2
        self.native_mpp = (self.lt_lat - self.rb_lat) * M_LAT / self.SH
        if kind == "png":
            res = self.native_mpp               # PNG is used at its own resolution
        self.res = res
        # resampled grid size (metres-square pixels)
        self.H = int(round((self.lt_lat - self.rb_lat) * M_LAT / res))
        self.W = int(round((self.rb_lon - self.lt_lon) * m_lng(lat_c) / res))
        if kind == "png":                       # use the PNG's own grid exactly (no resampling)
            self.H, self.W = self.SH, self.SW

    def latlon_to_px(self, lat, lon):
        x = (lon - self.lt_lon) / (self.rb_lon - self.lt_lon) * self.W
        y = (self.lt_lat - lat) / (self.lt_lat - self.rb_lat) * self.H
        return x, y

    def px_to_latlon(self, x, y):
        lon = self.lt_lon + x / self.W * (self.rb_lon - self.lt_lon)
        lat = self.lt_lat - y / self.H * (self.lt_lat - self.rb_lat)
        return lat, lon

    def read_tile(self, x0, y0, w, h):
        """Resampled-grid region -> RGB uint8 (w, h in resampled pixels)."""
        sx, sy = self.SW / self.W, self.SH / self.H
        c0, c1 = int(math.floor(x0 * sx)), int(math.ceil((x0 + w) * sx))
        r0, r1 = int(math.floor(y0 * sy)), int(math.ceil((y0 + h) * sy))
        c1, r1 = min(c1, self.SW), min(r1, self.SH)
        if c1 <= c0 or r1 <= r0:
            return None
        a = np.asarray(self.src[r0:r1, c0:c1])[:, :, :3]
        ow = max(1, int(round((c1 - c0) / sx)))
        oh = max(1, int(round((r1 - r0) / sy)))
        if (ow, oh) != (a.shape[1], a.shape[0]):
            a = cv2.resize(a, (ow, oh), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(a[:h, :w])   # rounding can add one row/col

    def global_features(self, net, cache_dir):
        tag = f"{self.flight}_{self.kind}_{self.res:.3f}"
        cp = cache_dir / f"mapfeat_{tag}.npz"
        if cp.exists():
            z = np.load(cp)
            return z["k"], z["d"]
        K, D = [], []
        t0 = time.time()
        ny, nx = math.ceil(self.H / TILE), math.ceil(self.W / TILE)
        for ty in range(ny):
            for tx in range(nx):
                t = self.read_tile(tx * TILE, ty * TILE, TILE, TILE)
                if t is None or t.std() < 3:     # outside raster / blank
                    continue
                k, s, d = net.tile_features(t)
                K.append(k + np.array([tx * TILE, ty * TILE], np.float32)); D.append(d)
            print(f"    map {tag}: tile row {ty + 1}/{ny} ({time.time() - t0:.0f}s)", end="\r", flush=True)
        K, D = np.concatenate(K), np.concatenate(D).astype(np.float16)
        np.savez(cp, k=K, d=D)
        print(f"\n    map {tag}: {len(K)} keypoints over {self.W}x{self.H} px in {time.time() - t0:.0f}s")
        return K, D


# ----------------------------------------------------------------------------- stills
def load_stills(flight):
    rows = list(csv.DictReader(open(DATA / flight / f"{flight}.csv", newline="")))
    agl = {}
    ap = CSV_DIR / f"agl_uavvisloc_{flight}.csv"
    if ap.exists():
        agl = {int(r["Num"]): float(r["AGL"]) for r in csv.DictReader(open(ap, newline=""))}
    out = []
    for i, r in enumerate(rows):
        out.append(dict(idx=i, file=r["filename"], lat=float(r["lat"]), lon=float(r["lon"]),
                        h=agl.get(int(r["num"]), float(r["height"])), hdg=float(r["Phi1"])))
    return out


def still_north_up(path, agl, hdg, res):
    """Read a still, scale to `res` m/px, rotate north-up. Returns RGB image, valid mask,
    and the centre's pixel position in the output."""
    img = cv2.imread(str(path))
    H0, W0 = img.shape[:2]
    gsd = 2 * agl * math.tan(math.radians(HFOV_DEG / 2)) / W0
    s = gsd / res
    small = cv2.resize(img, (max(1, int(W0 * s)), max(1, int(H0 * s))), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    D = int(math.ceil(math.hypot(w, h))) + 2
    # NORTH-UP NEEDS -Phi1 (OpenCV positive angle = counter-clockwise). Measured 2026-11-17 by an
    # angle sweep with the prior at the truth: still 320 of flight 01 gives 48 inliers at -Phi1
    # vs 7 at +Phi1; stills 0/700 agree. The C++ heading pre-warp (VideoProcessing.cpp) uses
    # +heading -- the opposite sign.
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -hdg, 1.0)
    M[0, 2] += D / 2 - w / 2
    M[1, 2] += D / 2 - h / 2
    rot = cv2.warpAffine(small, M, (D, D), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (D, D), flags=cv2.INTER_NEAREST)
    return rot[:, :, ::-1].copy(), valid, (D / 2, D / 2), gsd


# ----------------------------------------------------------------------------- main
def register_flight(flight, kind, res, step, net, cache_dir, seed, limit):
    mp = MapSource(flight, kind, res)
    res = mp.res
    MK, MD = mp.global_features(net, cache_dir)
    stills = load_stills(flight)
    rng = np.random.default_rng(seed + int(flight))
    win_px = WINDOW_M / res
    results = []
    t0 = time.time()
    sel = stills[::step][:limit] if limit else stills[::step]
    for n, st in enumerate(sel):
        off = rng.normal(0, SIGMA_M / math.sqrt(2), 2)            # (east, north) metres
        p_lat = st["lat"] + off[1] / M_LAT
        p_lon = st["lon"] + off[0] / m_lng(st["lat"])
        px, py = mp.latlon_to_px(p_lat, p_lon)
        rot, valid, (cx, cy), gsd = still_north_up(DATA / flight / "drone" / st["file"], st["h"], st["hdg"], res)
        half = rot.shape[0] / 2 + win_px                            # window half-size, map px
        m = (np.abs(MK[:, 0] - px) <= half) & (np.abs(MK[:, 1] - py) <= half)
        rec = dict(flight=flight, idx=st["idx"], prior_err=float(np.hypot(*off)), accepted=False)
        if m.sum() < 50:
            rec["reason"] = "few map keypoints in window"
            results.append(rec); continue
        sk, ss, sd = net.features(rot, valid)
        if len(sk) < 20:
            rec["reason"] = "few still keypoints"
            results.append(rec); continue
        # LightGlue works best on mostly-overlapping pairs (how the C++ pipeline uses it: 300 m
        # crop vs 300 m crop). Matching the still against the whole +-WINDOW_M search area gave
        # 0-16 matches even with a 3 m prior (2026-11-17), so slide a still-sized sub-window over
        # the search area at half overlap and keep the placement with the most RANSAC inliers.
        D = rot.shape[0]
        stride = D / 2
        offs = np.arange(-win_px, win_px + 1e-6, stride)
        if offs[-1] < win_px:
            offs = np.append(offs, win_px)
        best = None
        tot_matches = 0
        for oy in offs:
            for ox in offs:
                qx, qy = px + ox, py + oy                       # sub-window centre, map px
                hw = 0.6 * D                                    # slightly larger than the still
                mm = (np.abs(MK[:, 0] - qx) <= hw) & (np.abs(MK[:, 1] - qy) <= hw)
                if mm.sum() < 30:
                    continue
                wk = MK[mm] - np.array([qx - hw, qy - hw], np.float32)
                wd = MD[mm].astype(np.float32)
                i0, i1, sc = net.match(sk, sd, (rot.shape[1], rot.shape[0]), wk, wd, (2 * hw, 2 * hw))
                tot_matches += len(i0)
                if len(i0) < 6:
                    continue
                A, inl = cv2.estimateAffinePartial2D(sk[i0], wk[i1], method=cv2.RANSAC,
                                                     ransacReprojThreshold=RANSAC_PX, maxIters=5000,
                                                     confidence=0.999)
                if A is None:
                    continue
                n_in = int(inl.sum())
                if best is None or n_in > best[0]:
                    best = (n_in, A, qx - hw, qy - hw, len(i0))
        rec["matches"] = int(tot_matches)
        if best is None:
            rec["reason"] = "few matches"
            results.append(rec); continue
        ninl, A, ox0, oy0, nm = best
        rec["best_matches"] = int(nm)
        scale = math.hypot(A[0, 0], A[1, 0])
        rotd = math.degrees(math.atan2(A[1, 0], A[0, 0]))
        c = A @ np.array([cx, cy, 1.0])
        lat, lon = mp.px_to_latlon(c[0] + ox0, c[1] + oy0)
        err = math.hypot((lat - st["lat"]) * M_LAT, (lon - st["lon"]) * m_lng(st["lat"]))
        ok = ninl >= MIN_INLIERS and abs(scale - 1) <= MAX_SCALE_DEV and abs(rotd) <= MAX_ROT_DEG
        rec.update(inliers=ninl, scale=scale, rot_deg=rotd, err=err, accepted=bool(ok),
                   est_lat=lat, est_lon=lon, still_kpts=int(len(sk)))
        if not ok:
            rec["reason"] = "geometry check"
        results.append(rec)
        if (n + 1) % 10 == 0:
            acc = [r for r in results if r["accepted"]]
            med = np.median([r["err"] for r in acc]) if acc else float("nan")
            print(f"    {flight} {kind}@{res:.2f}: {n + 1}/{len(sel)} stills, accepted {len(acc)}, "
                  f"median err {med:.1f} m ({time.time() - t0:.0f}s)", flush=True)
    return results


def summarise(results, label):
    n = len(results)
    acc = [r for r in results if r["accepted"]]
    e = np.array([r["err"] for r in acc]) if acc else np.array([])
    alle = np.array([r["err"] for r in results if "err" in r])
    s = dict(label=label, n=n, accepted=len(acc), accept_rate=len(acc) / max(n, 1))
    if len(e):
        s.update(med=float(np.median(e)), mean=float(e.mean()), p90=float(np.percentile(e, 90)),
                 a10=float(np.mean(e <= 10)), a20=float(np.mean(e <= 20)), a50=float(np.mean(e <= 50)))
    # share of ALL stills localised within 20/50 m (accepted and correct)
    s["all_a20"] = float(np.sum(e <= 20) / max(n, 1)) if len(e) else 0.0
    s["all_a50"] = float(np.sum(e <= 50) / max(n, 1)) if len(e) else 0.0
    s["rejected_but_good_lt50"] = int(np.sum([(not r["accepted"]) and r.get("err", 1e9) <= 50 for r in results]))
    print(f"  {label}: n={n} accepted {len(acc)} ({s['accept_rate']*100:.0f}%)", end="")
    if len(e):
        print(f" | accepted err med {s['med']:.1f} m, mean {s['mean']:.1f}, p90 {s['p90']:.0f} | "
              f"A@10/20/50 (of accepted) {s['a10']*100:.0f}/{s['a20']*100:.0f}/{s['a50']*100:.0f}% | "
              f"within 20/50 m of ALL stills {s['all_a20']*100:.0f}/{s['all_a50']*100:.0f}%")
    else:
        print()
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flights", default="01,03")
    ap.add_argument("--map", choices=["png", "geotiff"], default="png")
    ap.add_argument("--res", type=float, default=0.6, help="m/px for --map geotiff")
    ap.add_argument("--step", type=int, default=8, help="use every Nth still")
    ap.add_argument("--limit", type=int, default=0, help="max stills per flight (0 = all sampled)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--cache", default=str(REPO / "models" / "c2f_cache"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    held = {"02", "05", "06", "11"}
    flights = [f.strip() for f in a.flights.split(",")]
    if held & set(flights):
        sys.exit("held-out flights are a one-time gate, not a Step-1 dev run")
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    net = DiskLightGlue(a.threads)
    allr, summ = [], []
    for f in flights:
        r = register_flight(f, a.map, a.res, a.step, net, cache, a.seed, a.limit)
        summ.append(summarise(r, f"flight {f} {a.map}"))
        allr += r
    summ.append(summarise(allr, f"ALL {a.map}"))
    if a.out:
        json.dump(dict(args=vars(a), summary=summ, results=allr), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
