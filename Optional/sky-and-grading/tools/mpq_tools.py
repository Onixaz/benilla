"""Small StormLib wrapper used by the optional-data builder.

StormLib is loaded from the user's machine.  This module never writes to a
client installation: candidates are copied or created below the requested
output directory and are verified after compaction.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple


MPQ_OPEN_READ_ONLY = 0x00000100
MPQ_FILE_COMPRESS = 0x00000200
MPQ_FILE_REPLACEEXISTING = 0x80000000
MPQ_COMPRESSION_ZLIB = 0x02
MAX_FILE_COUNT = 8192


def find_stormlib(explicit: Optional[Path]) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("STORMLIB_PATH")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path.cwd() / "StormLib.dll",
            Path(__file__).resolve().parent / "StormLib.dll",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "StormLib.dll was not found; pass --stormlib or set STORMLIB_PATH"
    )


class Storm:
    def __init__(self, library: Path):
        if os.name != "nt":
            raise OSError("this StormLib ctypes wrapper currently supports Windows")
        self.library = Path(library)
        self.dll = ctypes.WinDLL(str(self.library), use_last_error=True)
        handle = ctypes.c_void_p
        signatures = {
            "SFileOpenArchive": (
                ctypes.c_bool,
                [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(handle)],
            ),
            "SFileCreateArchive": (
                ctypes.c_bool,
                [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(handle)],
            ),
            "SFileSetMaxFileCount": (ctypes.c_bool, [handle, ctypes.c_uint32]),
            "SFileAddFileEx": (
                ctypes.c_bool,
                [handle, ctypes.c_wchar_p, ctypes.c_char_p, ctypes.c_uint32,
                 ctypes.c_uint32, ctypes.c_uint32],
            ),
            "SFileCompactArchive": (
                ctypes.c_bool,
                [handle, ctypes.c_char_p, ctypes.c_bool],
            ),
            "SFileOpenFileEx": (
                ctypes.c_bool,
                [handle, ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(handle)],
            ),
            "SFileGetFileSize": (
                ctypes.c_uint32,
                [handle, ctypes.POINTER(ctypes.c_uint32)],
            ),
            "SFileReadFile": (
                ctypes.c_bool,
                [handle, ctypes.c_void_p, ctypes.c_uint32,
                 ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p],
            ),
            "SFileCloseFile": (ctypes.c_bool, [handle]),
            "SFileCloseArchive": (ctypes.c_bool, [handle]),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(self.dll, name)
            function.restype = result
            function.argtypes = arguments

    @staticmethod
    def fail(message: str) -> None:
        error = ctypes.get_last_error()
        raise OSError(error, "%s (Win32 error %d)" % (message, error))

    def open(self, archive: Path, flags: int = 0) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        if not self.dll.SFileOpenArchive(str(archive), 0, flags, ctypes.byref(handle)):
            self.fail("SFileOpenArchive failed: %s" % archive)
        return handle

    def create(self, archive: Path) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        if not self.dll.SFileCreateArchive(
            str(archive), 0, MAX_FILE_COUNT, ctypes.byref(handle)
        ):
            self.fail("SFileCreateArchive failed: %s" % archive)
        return handle

    def close_archive(self, handle: ctypes.c_void_p) -> None:
        if handle.value and not self.dll.SFileCloseArchive(handle):
            self.fail("SFileCloseArchive failed")

    def read(self, archive: Path, member: str) -> Optional[bytes]:
        archive_handle = self.open(archive, MPQ_OPEN_READ_ONLY)
        try:
            file_handle = ctypes.c_void_p()
            encoded = member.encode("ascii")
            if not self.dll.SFileOpenFileEx(
                archive_handle, encoded, 0, ctypes.byref(file_handle)
            ):
                return None
            try:
                high = ctypes.c_uint32(0)
                size = self.dll.SFileGetFileSize(file_handle, ctypes.byref(high))
                if size == 0xFFFFFFFF or high.value:
                    self.fail("SFileGetFileSize failed: %s" % member)
                buffer = ctypes.create_string_buffer(size)
                received = ctypes.c_uint32(0)
                if not self.dll.SFileReadFile(
                    file_handle, buffer, size, ctypes.byref(received), None
                ):
                    self.fail("SFileReadFile failed: %s" % member)
                if received.value != size:
                    raise OSError("short MPQ read for %s" % member)
                return buffer.raw[:size]
            finally:
                if file_handle.value and not self.dll.SFileCloseFile(file_handle):
                    self.fail("SFileCloseFile failed: %s" % member)
        finally:
            self.close_archive(archive_handle)

    def pack(
        self,
        archive: Path,
        files: Iterable[Tuple[Path, str]],
        base_archive: Optional[Path] = None,
    ) -> None:
        archive.parent.mkdir(parents=True, exist_ok=True)
        if base_archive:
            shutil.copy2(base_archive, archive)
            os.chmod(archive, archive.stat().st_mode | stat.S_IWRITE)
            handle = self.open(archive)
        else:
            handle = self.create(archive)
        pairs = list(files)
        listfile_temp = None
        if not base_archive:
            listfile_temp = tempfile.TemporaryDirectory(prefix="benilla-listfile-")
            listfile_path = Path(listfile_temp.name) / "listfile.txt"
            listfile_path.write_text(
                "\r\n".join(member for _, member in pairs) + "\r\n(listfile)\r\n",
                encoding="ascii",
            )
            pairs.append((listfile_path, "(listfile)"))
        try:
            if not self.dll.SFileSetMaxFileCount(handle, MAX_FILE_COUNT):
                self.fail("SFileSetMaxFileCount failed")
            for disk_path, member in pairs:
                if not self.dll.SFileAddFileEx(
                    handle,
                    str(disk_path),
                    member.encode("ascii"),
                    MPQ_FILE_COMPRESS | MPQ_FILE_REPLACEEXISTING,
                    MPQ_COMPRESSION_ZLIB,
                    MPQ_COMPRESSION_ZLIB,
                ):
                    self.fail("SFileAddFileEx failed: %s" % member)
        finally:
            self.close_archive(handle)

        handle = self.open(archive)
        try:
            if not self.dll.SFileCompactArchive(handle, None, False):
                self.fail("SFileCompactArchive failed")
        finally:
            self.close_archive(handle)

        for disk_path, member in pairs:
            actual = self.read(archive, member)
            if actual != disk_path.read_bytes():
                raise RuntimeError("packed member failed verification: %s" % member)
        if listfile_temp is not None:
            listfile_temp.cleanup()


def mount_key(path: Path) -> Tuple[int, int, str]:
    name = path.name.casefold()
    if name == "patch.mpq":
        return (1, 0, name)
    numeric = re.fullmatch(r"patch-(\d+)\.mpq", name)
    if numeric:
        return (2, int(numeric.group(1)), name)
    letter = re.fullmatch(r"patch-(.)\.mpq", name)
    if letter:
        return (3, ord(letter.group(1)), name)
    return (0, 0, name)


def archives_in_mount_order(data_dir: Path) -> List[Path]:
    paths = [
        path for path in Path(data_dir).iterdir()
        if path.is_file() and path.suffix.casefold() == ".mpq"
    ]
    return sorted(paths, key=mount_key)


def copies_for(storm: Storm, data_dir: Path, member: str) -> List[Tuple[Path, bytes]]:
    result = []
    for archive in archives_in_mount_order(data_dir):
        payload = storm.read(archive, member)
        if payload is not None:
            result.append((archive, payload))
    return result


def winning_payloads(
    storm: Storm, data_dir: Path, names: Iterable[str]
) -> Tuple[Dict[str, bytes], Dict[str, str]]:
    payloads: Dict[str, bytes] = {}
    winners: Dict[str, str] = {}
    for name in names:
        copies = copies_for(storm, data_dir, "DBFilesClient\\" + name)
        if not copies:
            raise FileNotFoundError("no mounted copy of DBFilesClient\\%s" % name)
        archive, payload = copies[-1]
        payloads[name] = payload
        winners[name] = archive.name
    return payloads, winners
