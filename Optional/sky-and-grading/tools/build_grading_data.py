#!/usr/bin/env python3
"""Build MONKEY post-lane BLP2 LUT strips and MonkeyZoneGrade.dbc.

Outputs ordinary loose patch-tree files below this directory. Copy the resulting
DBFilesClient/ and World/ trees into an isolated Data directory for capture runs.
"""

from pathlib import Path
import math
import struct

ROOT = Path(__file__).resolve().parent.parent / "data"
LUT_DIR = ROOT / "World" / "LUTs"
DBC_DIR = ROOT / "DBFilesClient"


def clamp(value):
    return max(0.0, min(1.0, value))


def identity(c):
    return c


def elwynn(c):
    r, g, b = c
    return (clamp(r * 1.025 + 0.010 * g), clamp(g * 1.035 + 0.012 * r), clamp(b * 0.975))


def duskwood(c):
    r, g, b = c
    y = r * 0.299 + g * 0.587 + b * 0.114
    r, g, b = (y + (r - y) * 0.78, y + (g - y) * 0.78, y + (b - y) * 0.78)
    return (clamp(r * 0.94), clamp(g * 0.97), clamp(b * 1.045 + 0.008))


def westfall(c):
    r, g, b = c
    return (clamp(r * 1.045 + 0.008 * g), clamp(g * 1.018), clamp(b * 0.94))


def redridge(c):
    r, g, b = c
    return (clamp(r * 1.055 + 0.008), clamp(g * 0.982), clamp(b * 0.955))


def write_blp(path, transform):
    width, height = 1024, 32
    header_bytes = 148
    palette_bytes = 256 * 4
    pixel_offset = header_bytes + palette_bytes
    pixel_size = width * height * 4
    header = bytearray(header_bytes)
    header[0:4] = b"BLP2"
    struct.pack_into("<I", header, 4, 1)  # Direct content
    header[8:12] = bytes((3, 8, 0, 0))  # Raw3 BGRA8, no mip chain
    struct.pack_into("<II", header, 12, width, height)
    struct.pack_into("<I", header, 20, pixel_offset)
    struct.pack_into("<I", header, 84, pixel_size)

    pixels = bytearray()
    for g in range(32):
        for b in range(32):
            for r in range(32):
                source = (r / 31.0, g / 31.0, b / 31.0)
                rr, gg, bb = transform(source)
                rgba = [round(clamp(v) * 255.0) for v in (rr, gg, bb)]
                pixels.extend((rgba[2], rgba[1], rgba[0], 255))  # BLP Raw3 is BGRA
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + bytes(palette_bytes) + pixels)


def write_dbc(path):
    # ID, AreaID, DayLUT, NightLUT, Strength. Area IDs are the zone-level AreaTable rows.
    rows = [
        (1, 12, "World\\LUTs\\ElwynnWarmGreen.blp", "World\\LUTs\\Identity.blp", 0.34),
        (2, 10, "World\\LUTs\\Identity.blp", "World\\LUTs\\DuskwoodCold.blp", 0.44),
        (3, 40, "World\\LUTs\\WestfallGolden.blp", "World\\LUTs\\Identity.blp", 0.38),
        (4, 44, "World\\LUTs\\RedridgeWarmRed.blp", "World\\LUTs\\Identity.blp", 0.34),
    ]
    strings = bytearray(b"\0")
    offsets = {"": 0}

    def string_offset(value):
        if value not in offsets:
            offsets[value] = len(strings)
            strings.extend(value.encode("utf-8") + b"\0")
        return offsets[value]

    records = bytearray()
    for row_id, area_id, day, night, strength in rows:
        records.extend(struct.pack("<IIIIf", row_id, area_id, string_offset(day), string_offset(night), strength))
    header = b"WDBC" + struct.pack("<IIII", len(rows), 5, 20, len(strings))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + records + strings)


def main():
    for name, transform in [
        ("Identity.blp", identity),
        ("ElwynnWarmGreen.blp", elwynn),
        ("DuskwoodCold.blp", duskwood),
        ("WestfallGolden.blp", westfall),
        ("RedridgeWarmRed.blp", redridge),
    ]:
        write_blp(LUT_DIR / name, transform)
    write_dbc(DBC_DIR / "MonkeyZoneGrade.dbc")
    print(f"wrote 5 LUTs to {LUT_DIR}")
    print(f"wrote {DBC_DIR / 'MonkeyZoneGrade.dbc'}")


if __name__ == "__main__":
    main()
