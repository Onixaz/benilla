"""Moderated WoW Forever LightData colour import.

The modern table is read directly from the user's CASC installation.  Only
colour tracks are imported; fog distances, shadow opacity, and modern-only fog
fields remain untouched.
"""

from __future__ import annotations

import bisect
import math
import struct
from typing import Dict, Iterable, List, Tuple

from dbc_build import pack_dbc, parse_dbc


DAY = 2880
ANCHORS = (0, 720, 1440, 2160)
TARGET_PARAMS = (
    2, 3, 4, 10, 12, 13, 14, 15, 17, 21, 22, 23,
    24, 25, 192, 193, 206, 213, 269, 271, 454, 456, 494, 497,
)
DUSKWOOD_LOCAL = (14, 15, 494)
BLENDED_BANDS = tuple(index for index in range(18) if index != 8)


def unpack_band(row: List[int]) -> Tuple[List[int], List[int]]:
    count = row[1]
    if count > 16:
        raise ValueError("LightIntBand row has more than 16 keys")
    return row[2 : 2 + count], row[18 : 18 + count]


def make_band(record_id: int, times: List[int], values: List[int]) -> List[int]:
    if len(times) != len(values) or len(times) > 16:
        raise ValueError("invalid band key arrays")
    return [record_id, len(times)] + times + [0] * (16 - len(times)) + values + [0] * (
        16 - len(values)
    )


def rgb_tuple(value: int) -> Tuple[float, float, float]:
    return (
        ((value >> 16) & 255) / 255.0,
        ((value >> 8) & 255) / 255.0,
        (value & 255) / 255.0,
    )


def rgb_int(rgb: Iterable[float]) -> int:
    channels = [max(0, min(255, round(value * 255.0))) for value in rgb]
    return (channels[0] << 16) | (channels[1] << 8) | channels[2]


def sample(row: List[int], time: int) -> int:
    times, values = unpack_band(row)
    if not times:
        return 0
    if len(times) == 1:
        return values[0] & 0xFFFFFF
    time %= DAY
    if time < times[0] or time >= times[-1]:
        left_t, right_t = times[-1], times[0] + DAY
        left_v, right_v = values[-1], values[0]
        query = time + DAY if time < times[0] else time
    else:
        right = bisect.bisect_right(times, time)
        left_t, right_t = times[right - 1], times[right]
        left_v, right_v = values[right - 1], values[right]
        query = time
    amount = (query - left_t) / (right_t - left_t)
    return rgb_int(
        a + (b - a) * amount for a, b in zip(rgb_tuple(left_v), rgb_tuple(right_v))
    )


def srgb_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def linear_to_srgb(value: float) -> float:
    value = max(0.0, value)
    return 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055


def rgb_to_oklab(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    red, green, blue = (srgb_to_linear(value) for value in rgb)
    ll = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    mm = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    ss = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    ll, mm, ss = (
        math.copysign(abs(ll) ** (1 / 3), ll),
        math.copysign(abs(mm) ** (1 / 3), mm),
        math.copysign(abs(ss) ** (1 / 3), ss),
    )
    return (
        0.2104542553 * ll + 0.7936177850 * mm - 0.0040720468 * ss,
        1.9779984951 * ll - 2.4285922050 * mm + 0.4505937099 * ss,
        0.0259040371 * ll + 0.7827717662 * mm - 0.8086757660 * ss,
    )


def oklab_to_rgb(lab: Tuple[float, float, float]) -> Tuple[float, float, float]:
    light, aa, bb = lab
    ll = light + 0.3963377774 * aa + 0.2158037573 * bb
    mm = light - 0.1055613458 * aa - 0.0638541728 * bb
    ss = light - 0.0894841775 * aa - 1.2914855480 * bb
    ll, mm, ss = ll ** 3, mm ** 3, ss ** 3
    red = 4.0767416621 * ll - 3.3077115913 * mm + 0.2309699292 * ss
    green = -1.2684380046 * ll + 2.6097574011 * mm - 0.3413193965 * ss
    blue = -0.0041960863 * ll - 0.7034186147 * mm + 1.7076147010 * ss
    return tuple(
        max(0.0, min(1.0, linear_to_srgb(value))) for value in (red, green, blue)
    )


def chroma_hue(lab: Tuple[float, float, float]) -> Tuple[float, float]:
    _, aa, bb = lab
    return math.hypot(aa, bb), math.degrees(math.atan2(bb, aa)) % 360.0


def clamp_duskwood(
    base: Tuple[float, float, float], mixed: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    base_chroma, base_hue = chroma_hue(base)
    mixed_chroma, mixed_hue = chroma_hue(mixed)
    mixed_chroma = min(mixed_chroma, base_chroma + 0.015)
    if mixed_chroma < 1e-8:
        return (mixed[0], 0.0, 0.0)
    if base_chroma >= 0.006:
        shift = (mixed_hue - base_hue + 180.0) % 360.0 - 180.0
        hue = base_hue + max(-12.0, min(12.0, shift))
    else:
        hue = 155.0
    radians = math.radians(hue)
    return (
        mixed[0],
        mixed_chroma * math.cos(radians),
        mixed_chroma * math.sin(radians),
    )


def blend(base_value: int, wf_value: int, amount: float, duskwood: bool) -> int:
    if base_value == wf_value or amount <= 0.0:
        return base_value
    base = rgb_to_oklab(rgb_tuple(base_value))
    modern = rgb_to_oklab(rgb_tuple(wf_value))
    mixed = tuple(a + (b - a) * amount for a, b in zip(base, modern))
    if duskwood:
        mixed = clamp_duskwood(base, mixed)
    return rgb_int(oklab_to_rgb(mixed))


def strength(time: int, night: float, day: float) -> float:
    day_weight = (1.0 - math.cos(2.0 * math.pi * (time % DAY) / DAY)) * 0.5
    return night + (day - night) * day_weight


def modern_field_for_band(band: int) -> int:
    if band <= 7:
        return 3 + band
    if band == 8:
        return 20
    return band + 2


def apply_wf_colours(base_payload: bytes, light_data: object) -> Tuple[bytes, int]:
    records, strings, fields = parse_dbc(base_payload, "LightIntBand")
    if fields != 34 or strings != b"\0":
        raise ValueError("unexpected LightIntBand layout")
    by_id = {row[0]: row for row in records}
    modern: Dict[int, List[object]] = {}
    for row in light_data.rows.values():
        values = row["values"]
        param = row.get("parent_id", values[1])
        if param in TARGET_PARAMS:
            modern.setdefault(param, []).append(row)
    missing = [param for param in TARGET_PARAMS if not modern.get(param)]
    if missing:
        raise ValueError("modern LightData is missing target params: %s" % missing)

    changed = 0
    for param in TARGET_PARAMS:
        rows = sorted(
            modern[param], key=lambda row: (row["values"][2] & 0xFFFF, row["id"])
        )
        if len(rows) > 16:
            step = (len(rows) - 1) / 15.0
            rows = [rows[round(index * step)] for index in range(16)]
        for band in BLENDED_BANDS:
            record_id = (param - 1) * 18 + band + 1
            if record_id not in by_id:
                raise ValueError("missing LightIntBand row %d" % record_id)
            base_row = by_id[record_id]
            modern_times = [row["values"][2] & 0xFFFF for row in rows]
            field = modern_field_for_band(band)
            modern_values = [row["values"][field] & 0xFFFFFF for row in rows]
            modern_row = make_band(record_id, modern_times, modern_values)
            base_times, _ = unpack_band(base_row)
            times = sorted(set(base_times) | set(modern_times) | set(ANCHORS))
            if len(times) > 16:
                raise ValueError("moderated row %d needs more than 16 keys" % record_id)
            duskwood = param in DUSKWOOD_LOCAL
            night, day = ((0.30, 0.15) if duskwood else (0.50, 0.25))
            values = [
                blend(
                    sample(base_row, time),
                    sample(modern_row, time),
                    strength(time, night, day),
                    duskwood,
                )
                for time in times
            ]
            if any(sample(base_row, time) != value for time, value in zip(times, values)):
                by_id[record_id][:] = make_band(record_id, times, values)
                changed += 1
    return pack_dbc(records, strings, fields), changed
