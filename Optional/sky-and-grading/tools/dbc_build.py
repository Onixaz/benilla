"""Winner-safe lighting DBC merge for the optional sky patch."""

from __future__ import annotations

import struct
from typing import Dict, List, Set, Tuple


REQUIRED = (
    "Light.dbc",
    "LightParams.dbc",
    "LightIntBand.dbc",
    "LightFloatBand.dbc",
    "LightSkybox.dbc",
)


def parse_dbc(data: bytes, label: str) -> Tuple[List[List[int]], bytes, int]:
    if len(data) < 20:
        raise ValueError("truncated DBC: %s" % label)
    magic, count, fields, record_size, string_size = struct.unpack_from("<4s4I", data)
    if magic != b"WDBC" or record_size != fields * 4:
        raise ValueError("invalid WDBC schema: %s" % label)
    expected = 20 + count * record_size + string_size
    if len(data) != expected:
        raise ValueError("invalid WDBC length: %s" % label)
    records = [
        list(struct.unpack_from("<%dI" % fields, data, 20 + i * record_size))
        for i in range(count)
    ]
    return records, data[20 + count * record_size :], fields


def pack_dbc(records: List[List[int]], strings: bytes, fields: int) -> bytes:
    if any(len(record) != fields for record in records):
        raise ValueError("inconsistent DBC record width")
    header = struct.pack("<4s4I", b"WDBC", len(records), fields, fields * 4, len(strings))
    body = b"".join(struct.pack("<%dI" % fields, *record) for record in records)
    return header + body + strings


def indexed(records: List[List[int]], label: str) -> Dict[int, List[int]]:
    result = {record[0]: record for record in records}
    if len(result) != len(records):
        raise ValueError("duplicate row ID in %s" % label)
    return result


def band_row_id(param: int, band: int, per: int) -> int:
    return (param - 1) * per + band + 1


def append_string(block: bytearray, value: str) -> int:
    encoded = value.encode("utf-8") + b"\0"
    start = 0
    while True:
        offset = block.find(encoded, start)
        if offset < 0:
            offset = len(block)
            block.extend(encoded)
            return offset
        if offset == 0 or block[offset - 1] == 0:
            return offset
        start = offset + 1


def build_merged(winners: Dict[str, bytes]) -> Tuple[Dict[str, bytes], Dict[str, object]]:
    missing = [name for name in REQUIRED if name not in winners]
    if missing:
        raise ValueError("missing winning source DBCs: %s" % ", ".join(missing))

    light, light_strings, light_fields = parse_dbc(winners["Light.dbc"], "Light")
    params, params_strings, params_fields = parse_dbc(
        winners["LightParams.dbc"], "LightParams"
    )
    ints, int_strings, int_fields = parse_dbc(
        winners["LightIntBand.dbc"], "LightIntBand"
    )
    floats, float_strings, float_fields = parse_dbc(
        winners["LightFloatBand.dbc"], "LightFloatBand"
    )
    skies, sky_strings, sky_fields = parse_dbc(
        winners["LightSkybox.dbc"], "LightSkybox"
    )
    if (light_fields, params_fields, int_fields, float_fields, sky_fields) != (
        12, 9, 34, 34, 2
    ):
        raise ValueError("unexpected source lighting schema; use an unmodified 1.12 DBC chain")
    if any(block != b"\0" for block in (
        light_strings, params_strings, int_strings, float_strings
    )):
        raise ValueError("numeric lighting DBC has a non-empty string block")

    light_by_id = indexed(light, "Light")
    params_by_id = indexed(params, "LightParams")
    int_by_id = indexed(ints, "LightIntBand")
    float_by_id = indexed(floats, "LightFloatBand")

    targets = {20: 1139, 539: 1139, 6: 1140, 65: 1140, 67: 1140}
    expected = {20: 23, 539: 23, 6: 21, 65: 21, 67: 21}
    if light_by_id.get(14, [None] * 8)[7] != 16:
        raise ValueError("Light row 14 is not on expected Swamp of Sorrows param 16")
    for row_id, new_param in targets.items():
        if row_id not in light_by_id or light_by_id[row_id][7] != expected[row_id]:
            raise ValueError("unexpected Light[%d].Clear" % row_id)
        light_by_id[row_id][7] = new_param

    start_param = max(
        max(params_by_id),
        (max(int_by_id) + 17) // 18,
        (max(float_by_id) + 5) // 6,
    ) + 1
    if start_param != 1139:
        raise ValueError(
            "winner-derived clone start is %d, expected 1139; refusing an unsafe merge"
            % start_param
        )

    for source_param, new_param, skybox_id in ((23, 1139, 8), (21, 1140, 9)):
        clone = params_by_id[source_param][:]
        clone[0] = new_param
        clone[2] = skybox_id
        params.append(clone)
        for per, source, destination in (
            (18, int_by_id, ints),
            (6, float_by_id, floats),
        ):
            for band in range(per):
                source_id = band_row_id(source_param, band, per)
                target_id = band_row_id(new_param, band, per)
                if source_id not in source or target_id in source:
                    raise ValueError("unsafe band clone %d -> %d" % (source_id, target_id))
                row = source[source_id][:]
                row[0] = target_id
                destination.append(row)

    if params_by_id.get(269, [None, None, None])[2] != 0:
        raise ValueError("unexpected LightParams[269].LightSkybox")
    params_by_id[269][2] = 10

    strings = bytearray(sky_strings)
    extended = [[row[0], row[1], 0, 0] for row in skies]
    specs = (
        (8, r"Environments\Stars\10can_volcanicsky01.mdx", 15, ""),
        (9, r"Environments\Stars\bonewastesskybox.mdx", 4, ""),
        (
            10,
            r"Environments\Stars\8drk_darkshoreelune_sky01.mdx",
            2,
            r"Environments\Stars\hyjal_skybox.mdx",
        ),
    )
    for row_id, model, flags, celestial in specs:
        extended.append(
            [row_id, append_string(strings, model), flags, append_string(strings, celestial)]
        )

    light.sort(key=lambda row: row[0])
    params.sort(key=lambda row: row[0])
    ints.sort(key=lambda row: row[0])
    floats.sort(key=lambda row: row[0])
    extended.sort(key=lambda row: row[0])
    outputs = {
        "Light.dbc": pack_dbc(light, light_strings, 12),
        "LightParams.dbc": pack_dbc(params, params_strings, 9),
        "LightIntBand.dbc": pack_dbc(ints, int_strings, 34),
        "LightFloatBand.dbc": pack_dbc(floats, float_strings, 34),
        "LightSkybox.dbc": pack_dbc(extended, bytes(strings), 4),
    }
    report = {
        "source_rows": {
            name: len(parse_dbc(winners[name], name)[0]) for name in REQUIRED
        },
        "output_rows": {
            name: len(parse_dbc(payload, name)[0]) for name, payload in outputs.items()
        },
        "changed_light_rows": sorted(targets),
        "light_14_clear": 16,
        "clone_params": [1139, 1140],
    }
    return outputs, report


def check_bands(int_payload: bytes, float_payload: bytes) -> None:
    for label, payload, per in (
        ("LightIntBand", int_payload, 18),
        ("LightFloatBand", float_payload, 6),
    ):
        records, _, _ = parse_dbc(payload, label)
        rows = indexed(records, label)
        for source_param, target_param in ((23, 1139), (21, 1140)):
            for band in range(per):
                source_id = band_row_id(source_param, band, per)
                target_id = band_row_id(target_param, band, per)
                if target_id not in rows or rows[source_id][1:] != rows[target_id][1:]:
                    raise ValueError("%s clone mismatch at row %d" % (label, target_id))
        top = band_row_id(1140, per - 1, per)
        stray = [row_id for row_id in rows if row_id > top]
        if stray:
            raise ValueError("%s has rows above param 1140 span" % label)
