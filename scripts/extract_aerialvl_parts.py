#!/usr/bin/env python3
"""
Extract the complete files from a SUBSET of AerialVL's byte-split VAL.zip parts, without the
full 58 GB download.

AerialVL (huggingface.co/datasets/hmf21/AerialVL, CC BY 4.0) ships its sequence data as
images/VAL/VAL.zip.001..027: ONE ZIP64 archive cut into 2^31-byte pieces (not a multi-disk zip,
so every offset in the central directory is absolute). The central directory sits at the end,
in the last part (027). Any member whose bytes lie entirely inside a single downloaded part can
therefore be read directly: local header at (offset - part_base), then the deflated payload.

Known layout (parsed 2026-11-17): part 001 = geo_referenced_map/ (large + small GeoTIFF) +
639 frames of long_trajtr/2023-03-16-18-04-01; part 027 = the last 498 frames of
short_trajtr/2023-03-18-16-55-37. Frames are named @<UTC ms>@<lon>@<lat>@.png.

Parts 019-021 (2026-11-24): short_trajtr/2023-03-11-11-48-35 (922 frames, midday) and
short_trajtr/2023-03-16-16-58-43 (501 frames, 17:02 local) -- the av3/av4 fresh test.

Usage:
    python3 -I scripts/extract_aerialvl_parts.py \
        --parts-dir Datasets/AerialVL/val_zip_parts --out Datasets/AerialVL/extracted [--parts 19,20,21]
"""
import argparse
import collections
import struct
import sys
import zlib
from pathlib import Path, PurePosixPath

PART = 2 ** 31
LAST_PART = 27


def read_central_directory(last_part: Path):
    base = (LAST_PART - 1) * PART
    d = last_part.read_bytes()
    e = d.rfind(b"PK\x06\x06")  # ZIP64 end-of-central-directory record
    if e < 0:
        sys.exit("ZIP64 EOCD not found in the last part")
    cd_size, cd_off = struct.unpack("<QQ", d[e + 40:e + 56])
    cd = d[cd_off - base: cd_off - base + cd_size]
    ents, i = [], 0
    while i < len(cd) and cd[i:i + 4] == b"PK\x01\x02":
        meth, = struct.unpack("<H", cd[i + 10:i + 12])
        crc, csz, usz = struct.unpack("<III", cd[i + 16:i + 28])
        nl, el, cl = struct.unpack("<HHH", cd[i + 28:i + 34])
        off, = struct.unpack("<I", cd[i + 42:i + 46])
        name = cd[i + 46:i + 46 + nl].decode("utf-8", errors="replace")
        ex = cd[i + 46 + nl:i + 46 + nl + el]
        j = 0
        while j + 4 <= len(ex):
            hid, hl = struct.unpack("<HH", ex[j:j + 4])
            body, k = ex[j + 4:j + 4 + hl], 0
            if hid == 1:  # ZIP64 extended info: fields present only if the 32-bit one is saturated
                if usz == 0xFFFFFFFF:
                    usz, = struct.unpack("<Q", body[k:k + 8]); k += 8
                if csz == 0xFFFFFFFF:
                    csz, = struct.unpack("<Q", body[k:k + 8]); k += 8
                if off == 0xFFFFFFFF:
                    off, = struct.unpack("<Q", body[k:k + 8]); k += 8
            j += 4 + hl
        ents.append(dict(off=off, csz=csz, usz=usz, meth=meth, crc=crc, name=name))
        i += 46 + nl + el + cl
    return ents


def safe_relpath(name: str):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        return None
    return Path(*p.parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--parts", default=None,
                    help="comma-separated part numbers to extract from (e.g. 19,20,21); default: every part on disk. "
                         "The last part is always read for the central directory.")
    args = ap.parse_args()

    last = args.parts_dir / f"VAL.zip.{LAST_PART:03d}"
    ents = read_central_directory(last)
    have = {int(p.suffix[1:]): p for p in args.parts_dir.glob("VAL.zip.[0-9][0-9][0-9]")}
    if args.parts:
        want = {int(x) for x in args.parts.split(",")}
        have = {k: v for k, v in have.items() if k in want}
    print(f"{len(ents)} archive members; parts on disk: {sorted(have)}")

    done, skipped = collections.Counter(), collections.Counter()
    for e in ents:
        rel = safe_relpath(e["name"])
        if rel is None:
            print(f"  refusing unsafe member name: {e['name']!r}")
            continue
        if e["name"].endswith("/"):
            continue
        part = e["off"] // PART + 1
        top = str(rel.parent)
        if part not in have:
            skipped[top] += 1
            continue
        base = (part - 1) * PART
        with open(have[part], "rb") as f:
            f.seek(e["off"] - base)
            lh = f.read(30)
            if lh[:4] != b"PK\x03\x04":
                print(f"  bad local header: {e['name']}")
                continue
            nl, el = struct.unpack("<HH", lh[26:30])
            start = e["off"] - base + 30 + nl + el
            if start + e["csz"] > have[part].stat().st_size:
                skipped[top] += 1  # straddles into the next (missing or separate) part
                continue
            f.seek(start)
            payload = f.read(e["csz"])
        data = zlib.decompress(payload, -15) if e["meth"] == 8 else payload
        if len(data) != e["usz"] or (zlib.crc32(data) & 0xFFFFFFFF) != e["crc"]:
            print(f"  CRC/size mismatch, skipped: {e['name']}")
            continue
        dst = args.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        done[top] += 1

    print("extracted:")
    for k, v in sorted(done.items()):
        print(f"  {v:5d}  {k}")
    print(f"not available (part missing or straddling): {sum(skipped.values())} members")


if __name__ == "__main__":
    main()
