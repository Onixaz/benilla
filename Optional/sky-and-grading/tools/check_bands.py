#!/usr/bin/env python3
"""Check corrected Light band clone row IDs in loose files or an MPQ chain."""

import argparse
from pathlib import Path

from dbc_build import check_bands
from mpq_tools import Storm, copies_for, find_stormlib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="stage directory or client Data directory")
    parser.add_argument("--stormlib", type=Path)
    parser.add_argument("--stage", action="store_true", help="path is a loose stage tree")
    args = parser.parse_args()
    if args.stage:
        root = args.path / "DBFilesClient"
        integer = (root / "LightIntBand.dbc").read_bytes()
        floating = (root / "LightFloatBand.dbc").read_bytes()
    else:
        storm = Storm(find_stormlib(args.stormlib))
        payloads = []
        for name in ("LightIntBand.dbc", "LightFloatBand.dbc"):
            copies = copies_for(storm, args.path, "DBFilesClient\\" + name)
            if not copies:
                raise FileNotFoundError(name)
            payloads.append(copies[-1][1])
        integer, floating = payloads
    check_bands(integer, floating)
    print("PASS: clone rows use (P-1)*count + band + 1 and match params 23/21")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
