"""Small, read-only local CASC/TACT reader for the installed WoW builds.

The implementation deliberately indexes only the latest local v7 .idx file per
bucket and opens payloads on demand.  It never writes to the CASC installation.
"""
from __future__ import annotations

import hashlib
import pathlib
import struct
import zlib
from typing import Dict, Iterable, List, Optional, Tuple


class CascError(Exception):
    pass


class EncryptedBlock(CascError):
    def __init__(self, key_name: bytes):
        super().__init__("encrypted BLTE block; missing TACT key %s" % key_name.hex())
        self.key_name = key_name


def _u24be(data: bytes) -> int:
    return int.from_bytes(data, "big")


def _rotl32(x: int, n: int) -> int:
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def _salsa20_block(key: bytes, nonce8: bytes, counter: int) -> bytes:
    # Salsa20/20 with the 64-bit nonce/64-bit block-counter construction.
    if len(key) == 16:
        constants = b"expand 16-byte k"
        key = key + key
    elif len(key) == 32:
        constants = b"expand 32-byte k"
    else:
        raise CascError("Salsa20 key must contain 16 or 32 bytes")
    c = struct.unpack("<4I", constants)
    k = struct.unpack("<8I", key)
    n0, n1 = struct.unpack("<2I", nonce8)
    state = [c[0], k[0], k[1], k[2], k[3], c[1], n0, n1,
             counter & 0xFFFFFFFF, (counter >> 32) & 0xFFFFFFFF,
             c[2], k[4], k[5], k[6], k[7], c[3]]
    x = state[:]
    for _ in range(10):
        # Column round then row round (Salsa20 double-round).
        for a, b, c_, d in ((0, 4, 8, 12), (5, 9, 13, 1),
                             (10, 14, 2, 6), (15, 3, 7, 11),
                             (0, 1, 2, 3), (5, 6, 7, 4),
                             (10, 11, 8, 9), (15, 12, 13, 14)):
            x[b] ^= _rotl32((x[a] + x[d]) & 0xFFFFFFFF, 7)
            x[c_] ^= _rotl32((x[b] + x[a]) & 0xFFFFFFFF, 9)
            x[d] ^= _rotl32((x[c_] + x[b]) & 0xFFFFFFFF, 13)
            x[a] ^= _rotl32((x[d] + x[c_]) & 0xFFFFFFFF, 18)
    return struct.pack("<16I", *((x[i] + state[i]) & 0xFFFFFFFF for i in range(16)))


def salsa20_xor(data: bytes, key: bytes, nonce8: bytes) -> bytes:
    out = bytearray(len(data))
    for pos in range(0, len(data), 64):
        stream = _salsa20_block(key, nonce8, pos // 64)
        part = data[pos:pos + 64]
        for j, value in enumerate(part):
            out[pos + j] = value ^ stream[j]
    return bytes(out)


def load_tact_keys(path: pathlib.Path) -> Dict[bytes, bytes]:
    keys: Dict[bytes, bytes] = {}
    if not path.exists():
        return keys
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        bits = line.replace(";", " ").split()
        if len(bits) < 2:
            continue
        try:
            name, key = bytes.fromhex(bits[0]), bytes.fromhex(bits[1])
        except ValueError:
            continue
        if len(name) == 8 and len(key) in (16, 32):
            keys[name] = key
    return keys


class CascReader:
    """Read a product from one local modern Battle.net CASC installation."""

    def __init__(self, install: str, build_key: str,
                 listfile: Optional[str] = None,
                 tact_keys: Optional[str] = None):
        self.install = pathlib.Path(install)
        self.data_dir = self.install / "Data" / "data"
        self.config_dir = self.install / "Data" / "config"
        self.build_key = build_key.lower()
        self.config = self._read_config(self.build_key)
        self.keys = load_tact_keys(pathlib.Path(tact_keys)) if tact_keys else {}
        self.by_ekey9: Dict[bytes, Tuple[int, int, int]] = {}
        self._load_local_indices()
        self.names_by_fdid: Dict[int, str] = {}
        self.fdids_by_name: Dict[str, int] = {}
        if listfile:
            self._load_listfile(pathlib.Path(listfile))
        self.encoding_map: Dict[bytes, List[bytes]] = {}
        self.root_map: Dict[int, List[bytes]] = {}
        self.encoding_header = {}
        self.root_header = {}

    def _config_path(self, key: str) -> pathlib.Path:
        return self.config_dir / key[:2] / key[2:4] / key

    def _read_config(self, key: str) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        for raw in self._config_path(key).read_text(encoding="ascii").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            result[name.strip()] = value.strip().split()
        return result

    def _latest_indices(self) -> Iterable[pathlib.Path]:
        groups: Dict[str, pathlib.Path] = {}
        for path in self.data_dir.glob("*.idx"):
            bucket = path.name[:2].lower()
            old = groups.get(bucket)
            # Generation is encoded in the remaining filename hex.  The newest
            # index in each bucket is sufficient because local indices are full.
            if old is None or int(path.stem[2:], 16) > int(old.stem[2:], 16):
                groups[bucket] = path
        if len(groups) != 16:
            raise CascError("expected 16 local index buckets, found %d" % len(groups))
        return (groups["%02x" % i] for i in range(16))

    def _load_local_indices(self) -> None:
        for expected_bucket, path in enumerate(self._latest_indices()):
            data = path.read_bytes()
            if len(data) < 40:
                raise CascError("short local index " + str(path))
            header_len, = struct.unpack_from("<I", data, 0)
            version, bucket, extra, size_bytes, offset_bytes, key_bytes, checksum_bytes = \
                struct.unpack_from("<H6B", data, 8)
            if header_len != 16 or version != 7 or bucket != expected_bucket:
                raise CascError("unexpected v7 index header in " + str(path))
            if (extra, size_bytes, offset_bytes, key_bytes) != (0, 4, 5, 9):
                raise CascError("unsupported v7 index geometry in " + str(path))
            entries_len, = struct.unpack_from("<I", data, 32)
            entry_size = key_bytes + offset_bytes + size_bytes
            if entries_len % entry_size:
                raise CascError("misaligned index entries in " + str(path))
            pos = 40
            end = pos + entries_len
            if end > len(data):
                raise CascError("truncated index entries in " + str(path))
            while pos < end:
                key9 = data[pos:pos + key_bytes]
                packed = int.from_bytes(data[pos + key_bytes:pos + key_bytes + offset_bytes], "big")
                stored_size = int.from_bytes(
                    data[pos + key_bytes + offset_bytes:pos + entry_size], "little")
                archive = packed >> 30
                offset = packed & 0x3FFFFFFF
                self.by_ekey9[key9] = (archive, offset, stored_size)
                pos += entry_size

    def _load_listfile(self, path: pathlib.Path) -> None:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for raw in stream:
                try:
                    left, name = raw.rstrip("\r\n").split(";", 1)
                    fdid = int(left)
                except (ValueError, TypeError):
                    continue
                normalized = name.replace("\\", "/").lower()
                self.names_by_fdid[fdid] = name
                self.fdids_by_name[normalized] = fdid

    def _decode_blte_chunk(self, chunk: bytes, chunk_index: int) -> bytes:
        if not chunk:
            return b""
        mode, payload = chunk[0:1], chunk[1:]
        if mode == b"N":
            return payload
        if mode == b"Z":
            return zlib.decompress(payload)
        if mode == b"E":
            if len(payload) < 3:
                raise CascError("short encrypted BLTE block")
            nkey = payload[0]
            key_name = payload[1:1 + nkey][::-1]
            at = 1 + nkey
            niv = payload[at]
            iv = bytearray(payload[at + 1:at + 1 + niv])
            at += 1 + niv
            algorithm = payload[at:at + 1]
            ciphertext = payload[at + 1:]
            key = self.keys.get(key_name)
            if key is None:
                raise EncryptedBlock(key_name)
            if algorithm != b"S":
                raise CascError("unknown BLTE encryption algorithm %r" % algorithm)
            for i in range(4):
                if i < len(iv):
                    iv[i] ^= (chunk_index >> (8 * i)) & 0xFF
            nonce = bytes(iv[:8]).ljust(8, b"\0")
            clear = salsa20_xor(ciphertext, key, nonce)
            return self._decode_blte_chunk(clear, chunk_index)
        raise CascError("unknown BLTE block mode %r" % mode)

    def decode_blte(self, data: bytes) -> bytes:
        if data[:4] != b"BLTE":
            raise CascError("payload is not BLTE")
        header_size, = struct.unpack_from(">I", data, 4)
        if header_size == 0:
            return self._decode_blte_chunk(data[8:], 0)
        if header_size < 12 or header_size > len(data):
            raise CascError("invalid BLTE header size")
        flags = data[8]
        count = _u24be(data[9:12])
        if flags != 0x0F or header_size < 12 + count * 24:
            raise CascError("invalid BLTE chunk table")
        chunks = []
        pos = 12
        for _ in range(count):
            compressed, decompressed = struct.unpack_from(">II", data, pos)
            digest = data[pos + 8:pos + 24]
            chunks.append((compressed, decompressed, digest))
            pos += 24
        pos = header_size
        output = []
        for index, (compressed, decompressed, digest) in enumerate(chunks):
            raw = data[pos:pos + compressed]
            if len(raw) != compressed:
                raise CascError("truncated BLTE chunk")
            if hashlib.md5(raw).digest() != digest:
                raise CascError("BLTE chunk checksum mismatch")
            clear = self._decode_blte_chunk(raw, index)
            if len(clear) != decompressed:
                raise CascError("BLTE decompressed-size mismatch")
            output.append(clear)
            pos += compressed
        return b"".join(output)

    def open_ekey(self, ekey: bytes) -> bytes:
        if isinstance(ekey, str):
            ekey = bytes.fromhex(ekey)
        location = self.by_ekey9.get(ekey[:9])
        if location is None:
            raise CascError("encoding key is not present locally: " + ekey.hex())
        archive, offset, stored_size = location
        data_path = self.data_dir / ("data.%03d" % archive)
        with data_path.open("rb") as stream:
            stream.seek(offset)
            record = stream.read(stored_size)
        if len(record) != stored_size or len(record) < 30:
            raise CascError("short local CASC record for " + ekey.hex())
        # Local data header: reversed 16-byte (zero-extended) EKey, payload size,
        # flags/checksum fields; the encoded BLTE stream begins at byte 30.
        if record[30:34] != b"BLTE":
            raise CascError("local CASC record has no BLTE payload")
        return self.decode_blte(record[30:])

    def load_encoding(self) -> None:
        pair = self.config.get("encoding")
        if not pair or len(pair) < 2:
            raise CascError("build config has no encoding CKey/EKey pair")
        data = self.open_ekey(bytes.fromhex(pair[1]))
        self._parse_encoding(data)

    def _parse_encoding(self, data: bytes) -> None:
        if data[:2] != b"EN":
            raise CascError("bad encoding-file signature")
        # Header fields are big-endian on disk.
        version, ckey_size, ekey_size = data[2], data[3], data[4]
        cpage_kib, epage_kib = struct.unpack_from(">HH", data, 5)
        cpages, epages = struct.unpack_from(">II", data, 9)
        unknown = data[17]
        espec_size, = struct.unpack_from(">I", data, 18)
        self.encoding_header = dict(version=version, ckey_size=ckey_size,
                                    ekey_size=ekey_size, cpage_kib=cpage_kib,
                                    epage_kib=epage_kib, cpages=cpages,
                                    epages=epages, unknown=unknown,
                                    espec_size=espec_size,
                                    decoded_size=len(data))
        # The ESpec string block follows the 22-byte header. Each CKey page-index
        # entry is first-key+MD5; the fixed-size pages immediately follow it.
        pos = 22 + espec_size
        cpage_size = cpage_kib * 1024
        epage_size = epage_kib * 1024
        cindex_pos = pos
        cdata_pos = cindex_pos + cpages * (ckey_size + 16)
        mapping: Dict[bytes, List[bytes]] = {}
        for page in range(cpages):
            page_data = data[cdata_pos + page * cpage_size:cdata_pos + (page + 1) * cpage_size]
            at = 0
            while at < len(page_data):
                if at + 2 > len(page_data):
                    break
                count = int.from_bytes(page_data[at:at + 2], "little")
                at += 2
                if count == 0:
                    break
                need = 4 + ckey_size + count * ekey_size
                if at + need > len(page_data):
                    raise CascError("truncated encoding CKey page")
                _decoded_size = int.from_bytes(page_data[at:at + 4], "big")
                at += 4
                ckey = page_data[at:at + ckey_size]
                at += ckey_size
                ekeys = []
                for _ in range(count):
                    ekeys.append(page_data[at:at + ekey_size])
                    at += ekey_size
                mapping[ckey] = ekeys
        self.encoding_map = mapping

    def open_ckey(self, ckey: bytes) -> bytes:
        if isinstance(ckey, str):
            ckey = bytes.fromhex(ckey)
        if not self.encoding_map:
            self.load_encoding()
        ekeys = self.encoding_map.get(ckey)
        if not ekeys:
            raise CascError("content key absent from encoding file: " + ckey.hex())
        errors = []
        for ekey in ekeys:
            try:
                return self.open_ekey(ekey)
            except CascError as exc:
                errors.append(str(exc))
        raise CascError("no local encoding for CKey %s: %s" % (ckey.hex(), "; ".join(errors)))

    def load_root(self) -> None:
        roots = self.config.get("root")
        if not roots:
            raise CascError("build config has no root CKey")
        data = self.open_ckey(bytes.fromhex(roots[0]))
        self._parse_root(data)

    def _parse_root(self, data: bytes) -> None:
        if data[:4] != b"TSFM" or len(data) < 24:
            raise CascError("unsupported root signature")
        header_size, version, total_files, named_files, flags = \
            struct.unpack_from("<5I", data, 4)
        if header_size != 24 or version != 2:
            raise CascError("unsupported TSFM root version/header")
        mapping: Dict[int, List[bytes]] = {}
        pos = header_size
        seen_total = 0
        seen_named = 0
        blocks = 0
        while pos < len(data):
            if pos + 17 > len(data):
                raise CascError("truncated TSFM block header")
            count, content_flags, locale_flags = struct.unpack_from("<III", data, pos)
            block_meta = data[pos + 12:pos + 17]
            if count == 0 or block_meta[:3] != b"\0\0\0":
                raise CascError("invalid TSFM block at offset %d" % pos)
            delta_pos = pos + 17
            ckey_pos = delta_pos + count * 4
            hash_pos = ckey_pos + count * 16
            # In TSFM v2 bit 0x10 of the fourth metadata byte marks entries
            # without filename hashes. This agrees with the root header's named
            # count and is also how the manifest stays compact for unnamed FDIDs.
            has_name_hash = (block_meta[3] & 0x10) == 0
            end = hash_pos + (count * 8 if has_name_hash else 0)
            if end > len(data):
                raise CascError("truncated TSFM block data")
            fdid = -1
            for index in range(count):
                delta, = struct.unpack_from("<I", data, delta_pos + index * 4)
                fdid += delta + 1
                ckey = data[ckey_pos + index * 16:ckey_pos + (index + 1) * 16]
                values = mapping.setdefault(fdid, [])
                if ckey not in values:
                    values.append(ckey)
            seen_total += count
            if has_name_hash:
                seen_named += count
            blocks += 1
            pos = end
        if seen_total != total_files or seen_named != named_files:
            raise CascError("TSFM record-count mismatch")
        self.root_header = dict(header_size=header_size, version=version,
                                total_files=total_files, named_files=named_files,
                                flags=flags, blocks=blocks)
        self.root_map = mapping

    def open_fdid(self, fdid: int) -> bytes:
        if not self.root_map:
            self.load_root()
        ckeys = self.root_map.get(int(fdid))
        if not ckeys:
            raise CascError("FileDataID absent from root: %d" % fdid)
        errors = []
        for ckey in ckeys:
            try:
                return self.open_ckey(ckey)
            except CascError as exc:
                errors.append(str(exc))
        raise CascError("no readable root entry for FDID %d: %s" % (fdid, "; ".join(errors)))

    def open_name(self, path: str) -> bytes:
        normalized = path.replace("\\", "/").lower()
        fdid = self.fdids_by_name.get(normalized)
        if fdid is None:
            raise CascError("path absent from listfile: " + path)
        return self.open_fdid(fdid)
