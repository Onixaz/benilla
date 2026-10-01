#!/usr/bin/env python3
"""Build an asset-free optional sky and colour-grading MPQ from user installs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

from casc import CascReader
from dbc_build import REQUIRED, build_merged, check_bands
from mpq_tools import Storm, find_stormlib, winning_payloads
from sky_convert import convert_all, extract_sources, load_pywowlib, read_listfile
from wdc import WDC5
from wf_colours import apply_wf_colours


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
LISTFILE_URL = (
    "https://github.com/wowdev/wow-listfile/releases/latest/download/"
    "community-listfile.csv"
)
TACT_KEYS_URL = "https://raw.githubusercontent.com/wowdev/TACTKeys/master/WoW.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, target: Path) -> Path:
    if target.is_file() and target.stat().st_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, 1024 * 1024)
    partial.replace(target)
    return target


def casc_root(install: Path) -> Path:
    install = Path(install).resolve()
    if (install / "Data" / "config").is_dir():
        return install
    if (install.parent / "Data" / "config").is_dir():
        return install.parent
    raise FileNotFoundError("no shared CASC Data/config below %s or its parent" % install)


def select_build_key(install: Path, preferred: str | None = None) -> tuple[str, str]:
    info = install / ".build.info"
    if not info.is_file():
        raise FileNotFoundError(info)
    with info.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="|")
        raw_headers = next(reader)
        headers = [header.split("!", 1)[0] for header in raw_headers]
        rows = [dict(zip(headers, row)) for row in reader]
    active = [row for row in rows if row.get("Active") == "1"] or rows
    products = [preferred] if preferred else []
    products.extend(name for name in ("wow_classic_beta", "wow") if name != preferred)
    for wanted in products:
        for row in active:
            if row.get("Product") == wanted:
                key = row.get("Build Key", "").lower()
                if len(key) == 32:
                    return key, wanted
    raise RuntimeError("no active WoW Forever/classic-beta or retail build in .build.info")


def reset_owned_directory(path: Path, parent: Path, force: bool) -> None:
    path = path.resolve()
    parent = parent.resolve()
    if path.parent != parent:
        raise RuntimeError("refusing to reset unexpected path %s" % path)
    if path.exists():
        if not force:
            raise FileExistsError("build directory exists; pass --force: %s" % path)
        shutil.rmtree(path)
    path.mkdir(parents=True)


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def copy_own_grading(stage: Path) -> None:
    for source in sorted(DATA.rglob("*")):
        if not source.is_file() or source.name == "LICENSE.md":
            continue
        relative = source.relative_to(DATA)
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def archive_base(client_data: Path, letter: str) -> Path | None:
    wanted = ("patch-%s.mpq" % letter).casefold()
    matches = [
        path for path in client_data.iterdir()
        if path.is_file() and path.name.casefold() == wanted
    ]
    if len(matches) > 1:
        raise RuntimeError("multiple case-insensitive matches for %s" % wanted)
    return matches[0] if matches else None


def stage_pairs(stage: Path) -> list[tuple[Path, str]]:
    return [
        (path, str(path.relative_to(stage)).replace("/", "\\"))
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wow-forever", type=Path, help="WoW Forever or retail install root")
    parser.add_argument("--client-data", type=Path, required=True, help="1.12.1 Data directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--grading-only", action="store_true", help="pack only project-owned grading data")
    parser.add_argument("--wf-colours", action="store_true", help="moderate locally extracted WoW Forever night colours")
    parser.add_argument("--pywowlib", type=Path, help="pinned external pywowlib checkout")
    parser.add_argument("--stormlib", type=Path, help="path to StormLib.dll")
    parser.add_argument("--listfile", type=Path, help="existing community-listfile.csv")
    parser.add_argument("--tact-keys", type=Path, help="existing public WoW TACT key list")
    parser.add_argument("--patch-letter", default="Z", help="single patch character (default: Z)")
    parser.add_argument("--force", action="store_true", help="replace an existing output archive")
    args = parser.parse_args()
    if len(args.patch_letter) != 1 or not args.patch_letter.isalpha():
        parser.error("--patch-letter must be one ASCII letter")
    if not args.client_data.is_dir():
        parser.error("--client-data is not a directory")
    if args.wf_colours and args.grading_only:
        parser.error("--wf-colours cannot be combined with --grading-only")
    if not args.grading_only and (not args.wow_forever or not args.pywowlib):
        parser.error("sky builds require --wow-forever and --pywowlib")
    return args


def main() -> int:
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or newer is required")
    args = parse_args()
    output = args.out.resolve()
    client_data = args.client_data.resolve()
    if is_within(output, client_data):
        raise ValueError("--out must not be inside the 1.12.1 Data directory")
    if args.wow_forever:
        modern_input = casc_root(args.wow_forever)
        if is_within(output, modern_input):
            raise ValueError("--out must not be inside the modern WoW installation")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / ("patch-%s.MPQ" % args.patch_letter.upper())
    if archive.exists() and not args.force:
        raise FileExistsError("output archive exists; pass --force: %s" % archive)
    stage, work = output / "stage", output / "work"
    reset_owned_directory(stage, output, args.force)
    reset_owned_directory(work, output, args.force)
    copy_own_grading(stage)

    storm = Storm(find_stormlib(args.stormlib))
    report: dict[str, object] = {"grading_only": args.grading_only}
    if not args.grading_only:
        cache = output / "cache"
        listfile_path = args.listfile or fetch(LISTFILE_URL, cache / "community-listfile.csv")
        tact_keys = args.tact_keys or fetch(TACT_KEYS_URL, cache / "WoW.txt")
        install = modern_input
        requested_name = args.wow_forever.resolve().name.casefold()
        preferred = {
            "_classic_beta_": "wow_classic_beta",
            "_retail_": "wow",
        }.get(requested_name)
        build_key, product = select_build_key(install, preferred)
        reader = CascReader(str(install), build_key, str(listfile_path), str(tact_keys))
        reader.load_encoding()
        reader.load_root()
        modules = load_pywowlib(args.pywowlib)
        source_root = work / "source"
        extract_sources(reader, read_listfile(listfile_path), source_root, modules)
        conversions = convert_all(source_root, stage, listfile_path, modules)

        winners, winner_names = winning_payloads(storm, args.client_data, REQUIRED)
        wf_changed = 0
        if args.wf_colours:
            light_data = WDC5(reader.open_fdid(1375580))
            winners["LightIntBand.dbc"], wf_changed = apply_wf_colours(
                winners["LightIntBand.dbc"], light_data
            )
        merged, dbc_report = build_merged(winners)
        dbc_dir = stage / "DBFilesClient"
        dbc_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in merged.items():
            (dbc_dir / name).write_bytes(payload)
        check_bands(merged["LightIntBand.dbc"], merged["LightFloatBand.dbc"])
        report.update(
            {
                "modern_product": product,
                "build_key": build_key,
                "dbc_winners": winner_names,
                "dbc_merge": dbc_report,
                "wf_colour_rows_changed": wf_changed,
                "conversions": conversions,
            }
        )

    if archive.exists():
        archive.unlink()
    base = archive_base(args.client_data, args.patch_letter)
    storm.pack(archive, stage_pairs(stage), base_archive=base)

    files = {
        str(path.relative_to(stage)).replace("/", "\\"): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    }
    report.update(
        {
            "archive": str(archive),
            "archive_base": str(base) if base else None,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256(archive),
            "stage_files": files,
        }
    )
    (output / "build-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "built %s with %d staged files%s"
        % (archive, len(files), " (WoW Forever colours)" if args.wf_colours else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
