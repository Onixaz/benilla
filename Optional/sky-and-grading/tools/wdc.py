"""Minimal WDC5 reader for dense modern WoW DB2 tables."""
from __future__ import annotations

import copy
import struct


class WdcError(Exception):
    pass


def _bits_le(data: bytes, bit_offset: int, bit_count: int) -> int:
    if bit_count == 0:
        return 0
    start = bit_offset // 8
    shift = bit_offset & 7
    count = (shift + bit_count + 7) // 8
    value = int.from_bytes(data[start:start + count], "little") >> shift
    return value & ((1 << bit_count) - 1)


class WDC5:
    HEADER_OFFSET = 0x88

    def __init__(self, data: bytes):
        self.data = data
        if data[:4] != b"WDC5" or len(data) < self.HEADER_OFFSET + 68:
            raise WdcError("not a supported WDC5 table")
        o = self.HEADER_OFFSET
        (self.record_count, self.field_count, self.record_size,
         self.string_table_size, self.table_hash, self.layout_hash,
         self.min_id, self.max_id, self.locale) = struct.unpack_from("<9I", data, o)
        self.flags, self.id_index = struct.unpack_from("<HH", data, o + 36)
        (self.total_field_count, self.bitpacked_data_offset,
         self.lookup_column_count, self.field_storage_info_size,
         self.common_data_size, self.pallet_data_size,
         self.section_count) = struct.unpack_from("<7I", data, o + 40)
        if self.section_count != 1:
            raise WdcError("only single-section WDC5 tables are supported")
        at = o + 68
        section = struct.unpack_from("<Q8I", data, at)
        (self.tact_key_hash, self.file_offset, section_records,
         self.section_string_size, self.offset_records_end,
         self.id_list_size, self.relationship_data_size,
         self.offset_map_id_count, self.copy_table_count) = section
        if section_records != self.record_count:
            raise WdcError("multi-section record count mismatch")
        at += 40
        self.field_structures = [struct.unpack_from("<hH", data, at + i * 4)
                                 for i in range(self.total_field_count)]
        at += self.total_field_count * 4
        self.storage = [struct.unpack_from("<HH5I", data, at + i * 24)
                        for i in range(self.total_field_count)]
        at += self.field_storage_info_size
        self.pallet = data[at:at + self.pallet_data_size]
        at += self.pallet_data_size
        self.common = data[at:at + self.common_data_size]
        at += self.common_data_size
        if at != self.file_offset:
            raise WdcError("WDC5 metadata/file-offset mismatch")
        self.records_offset = self.file_offset
        self.strings_offset = self.records_offset + self.record_count * self.record_size
        after_strings = self.strings_offset + self.section_string_size
        self.ids_offset = after_strings
        after_ids = self.ids_offset + self.id_list_size
        self.copy_offset = after_ids + self.relationship_data_size

        poff = coff = 0
        self.aux_offsets = []
        for _field_offset, _field_size, additional, kind, _a, _b, _c in self.storage:
            if kind in (3, 4):
                self.aux_offsets.append((poff, None))
                poff += additional
            elif kind == 2:
                self.aux_offsets.append((None, coff))
                coff += additional
            else:
                self.aux_offsets.append((None, None))
        if poff != self.pallet_data_size or coff != self.common_data_size:
            raise WdcError("WDC5 auxiliary-data size mismatch")

        if self.id_list_size:
            if self.id_list_size != self.record_count * 4:
                raise WdcError("unexpected WDC5 ID-list size")
            self.ids = list(struct.unpack_from("<%dI" % self.record_count,
                                               data, self.ids_offset))
        else:
            self.ids = [None] * self.record_count
        self.rows = {}
        self.row_ids = []
        for index in range(self.record_count):
            recpos = self.records_offset + index * self.record_size
            record = data[recpos:recpos + self.record_size]
            values = [self._value(index, record, i) for i in range(self.total_field_count)]
            row_id = self.ids[index]
            if row_id is None:
                row_id = values[self.id_index]
            self.rows[row_id] = {"id": row_id, "values": values,
                                 "record_offset": recpos, "copy_of": None}
            self.row_ids.append(row_id)
        self.relationship_by_index = {}
        if self.relationship_data_size:
            relpos = after_ids
            if self.relationship_data_size < 12:
                raise WdcError("short WDC5 relationship block")
            count, self.relationship_min_id, self.relationship_max_id = \
                struct.unpack_from("<3I", data, relpos)
            if self.relationship_data_size != 12 + count * 8:
                raise WdcError("unexpected WDC5 relationship-block size")
            for i in range(count):
                foreign_id, record_index = struct.unpack_from("<2I", data, relpos + 12 + i * 8)
                if record_index >= self.record_count:
                    raise WdcError("WDC5 relationship record index is out of bounds")
                self.relationship_by_index[record_index] = foreign_id
                self.rows[self.row_ids[record_index]]["parent_id"] = foreign_id
        for i in range(self.copy_table_count):
            new_id, source_id = struct.unpack_from("<II", data, self.copy_offset + i * 8)
            source = self.rows.get(source_id)
            if source is None:
                raise WdcError("copy-table source ID is absent")
            row = copy.deepcopy(source)
            row["id"] = new_id
            row["copy_of"] = source_id
            self.rows[new_id] = row

    def _value(self, row_index: int, record: bytes, field: int):
        bit_offset, bit_size, additional, kind, a, b, c = self.storage[field]
        raw = _bits_le(record, bit_offset, bit_size)
        if kind in (0, 1):
            return raw
        if kind == 5:
            if bit_size and raw & (1 << (bit_size - 1)):
                raw -= 1 << bit_size
            return raw
        if kind == 2:
            default = a
            _, coff = self.aux_offsets[field]
            block = self.common[coff:coff + additional]
            row_id = self.ids[row_index]
            for pos in range(0, len(block), 8):
                key, value = struct.unpack_from("<II", block, pos)
                if key == row_id:
                    return value
            return default
        if kind == 3:
            poff, _ = self.aux_offsets[field]
            if raw * 4 + 4 > additional:
                raise WdcError("palette index out of bounds")
            return struct.unpack_from("<I", self.pallet, poff + raw * 4)[0]
        if kind == 4:
            width = c
            poff, _ = self.aux_offsets[field]
            base = poff + raw * width * 4
            if base + width * 4 > poff + additional:
                raise WdcError("palette-array index out of bounds")
            return struct.unpack_from("<%dI" % width, self.pallet, base)
        raise WdcError("unsupported WDC5 storage type %d" % kind)

    def string(self, row, field: int) -> str:
        relative = row["values"][field]
        start = row["record_offset"] + relative
        if start < self.strings_offset or start >= self.strings_offset + self.section_string_size:
            return ""
        end = self.data.find(b"\0", start, self.strings_offset + self.section_string_size)
        if end < 0:
            end = self.strings_offset + self.section_string_size
        return self.data[start:end].decode("utf-8", errors="replace")

    @staticmethod
    def as_float(value: int) -> float:
        return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]
