#!/usr/bin/env python3
"""Verify every staged payload is the mounted winner in a candidate Data chain."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from dbc_build import check_bands, indexed, parse_dbc
from mpq_tools import Storm, copies_for, find_stormlib


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def semantic_checks(payloads: dict[str, bytes]) -> None:
    light, _, fields = parse_dbc(payloads["DBFilesClient\\Light.dbc"], "Light")
    if fields != 12:
        raise ValueError("Light.dbc is not 12 fields")
    light_rows = indexed(light, "Light")
    expected = {20: 1139, 539: 1139, 6: 1140, 65: 1140, 67: 1140, 14: 16}
    for row_id, clear in expected.items():
        if light_rows[row_id][7] != clear:
            raise ValueError("Light[%d].Clear is not %d" % (row_id, clear))
    params, _, fields = parse_dbc(
        payloads["DBFilesClient\\LightParams.dbc"], "LightParams"
    )
    if fields != 9:
        raise ValueError("LightParams.dbc is not 9 fields")
    param_rows = indexed(params, "LightParams")
    for row_id, skybox in ((1139, 8), (1140, 9), (269, 10)):
        if param_rows[row_id][2] != skybox:
            raise ValueError("LightParams[%d] skybox mismatch" % row_id)
    if 1141 in param_rows:
        raise ValueError("stray LightParams 1141")
    check_bands(
        payloads["DBFilesClient\\LightIntBand.dbc"],
        payloads["DBFilesClient\\LightFloatBand.dbc"],
    )
    skies, strings, fields = parse_dbc(
        payloads["DBFilesClient\\LightSkybox.dbc"], "LightSkybox"
    )
    if fields != 4 or not strings:
        raise ValueError("LightSkybox.dbc is not the extended four-field layout")
    sky_rows = indexed(skies, "LightSkybox")
    for row_id in (8, 9, 10):
        if row_id not in sky_rows:
            raise ValueError("LightSkybox row %d is missing" % row_id)
    grade, _, fields = parse_dbc(
        payloads["DBFilesClient\\MonkeyZoneGrade.dbc"], "MonkeyZoneGrade"
    )
    if fields != 5 or len(grade) != 4:
        raise ValueError("MonkeyZoneGrade must have four five-field rows")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--stormlib", type=Path)
    parser.add_argument("--require-winner", default="patch-Z.mpq")
    args = parser.parse_args()
    storm = Storm(find_stormlib(args.stormlib))
    expected = {
        str(path.relative_to(args.stage)).replace("/", "\\"): path.read_bytes()
        for path in sorted(args.stage.rglob("*"))
        if path.is_file()
    }
    mounted = {}
    failures = []
    for member, wanted in expected.items():
        copies = copies_for(storm, args.data_dir, member)
        if not copies:
            failures.append("missing %s" % member)
            continue
        archive, actual = copies[-1]
        mounted[member] = actual
        if args.require_winner.casefold() != "any" and (
            archive.name.casefold() != args.require_winner.casefold()
        ):
            failures.append("%s wins from %s" % (member, archive.name))
        if actual != wanted:
            failures.append(
                "%s differs: %s != %s" % (member, digest(actual), digest(wanted))
            )
    required = {
        "DBFilesClient\\Light.dbc",
        "DBFilesClient\\LightParams.dbc",
        "DBFilesClient\\LightIntBand.dbc",
        "DBFilesClient\\LightFloatBand.dbc",
        "DBFilesClient\\LightSkybox.dbc",
        "DBFilesClient\\MonkeyZoneGrade.dbc",
    }
    if required <= set(mounted):
        try:
            semantic_checks(mounted)
        except (KeyError, ValueError) as error:
            failures.append("semantic check: %s" % error)
    if failures:
        print("FAIL")
        for failure in failures:
            print("  " + failure)
        return 1
    print("PASS: %d staged payloads are mounted winners" % len(expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
