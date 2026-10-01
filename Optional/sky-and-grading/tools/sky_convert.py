"""Extract four modern sky models from local CASC and convert them to M2 v256.

pywowlib is an external dependency.  The conversion and v256 compatibility
fixes here are original glue; no pywowlib source is redistributed.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from typing import Dict, Iterable, List, Tuple


PYWOWLIB_COMMIT = "55276dc5c2195da7fe136638a2a59716622f8c65"
MODELS = (
    ("blasted_lands", 130481, "bonewastesskybox.m2"),
    ("mount_hyjal_main", 2322316, "8drk_darkshoreelune_sky01.m2"),
    ("mount_hyjal_celestial", 317347, "hyjal_skybox.m2"),
    ("burning_steppes", 4505901, "10can_volcanicsky01.m2"),
)


def load_pywowlib(checkout: Path) -> Dict[str, object]:
    checkout = Path(checkout).resolve()
    package = checkout if checkout.name.casefold() == "pywowlib" else checkout / "pywowlib"
    if not (package / "m2_file.py").is_file():
        raise FileNotFoundError("--pywowlib must name a pywowlib checkout")
    try:
        head = subprocess.check_output(
            ["git", "-C", str(package), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("the pywowlib checkout must retain its .git metadata") from error
    if head != PYWOWLIB_COMMIT:
        raise RuntimeError(
            "pywowlib HEAD is %s; expected pinned commit %s" % (head, PYWOWLIB_COMMIT)
        )
    sys.path.insert(0, str(package.parent))
    modules = {
        "common": importlib.import_module("pywowlib.file_formats.wow_common_types"),
        "types": importlib.import_module("pywowlib.io_utils.types"),
        "m2_file": importlib.import_module("pywowlib.m2_file"),
        "m2_format": importlib.import_module("pywowlib.file_formats.m2_format"),
        "m2_chunks": importlib.import_module("pywowlib.file_formats.m2_chunks"),
        "skin_format": importlib.import_module("pywowlib.file_formats.skin_format"),
    }
    return modules


def read_listfile(path: Path) -> Dict[int, str]:
    result = {}
    with Path(path).open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            raw_id, separator, name = line.rstrip("\r\n").partition(";")
            if separator and raw_id.isdigit():
                result[int(raw_id)] = name
    return result


def source_version(path: Path) -> int:
    data = path.read_bytes()[:16]
    if data[:4] == b"MD20":
        return struct.unpack_from("<I", data, 4)[0]
    if data[:4] == b"MD21" and data[8:12] == b"MD20":
        return struct.unpack_from("<I", data, 12)[0]
    raise ValueError("unable to determine M2 version: %s" % path)


def expansion_for(version: int) -> int:
    versions = {256: 0, 263: 1, 264: 2, 272: 3, 273: 5, 274: 6}
    if version not in versions:
        raise ValueError("unsupported M2 version %d" % version)
    return versions[version]


def extract_sources(reader: object, listfile: Dict[int, str], root: Path, modules: Dict[str, object]) -> None:
    M2File = modules["m2_file"].M2File
    manager = modules["common"].M2VersionsManager()
    for _, file_id, output_name in MODELS:
        source_dir = root / str(file_id)
        source_dir.mkdir(parents=True, exist_ok=True)
        model_path = source_dir / output_name
        model_path.write_bytes(reader.open_fdid(file_id))
        version = source_version(model_path)
        manager.set_m2_version(modules["common"].M2Versions.from_expansion_number(expansion_for(version)))
        model = M2File(expansion_for(version), filepath=str(model_path))
        dependencies = model.find_model_dependencies()
        ids = set(int(value) for value in dependencies.skins)
        for index, texture in enumerate(model.root.textures):
            texture_id = int(getattr(texture, "fdid", 0) or 0)
            if not texture_id and model.txid and index < len(model.txid.texture_ids):
                texture_id = int(model.txid.texture_ids[index])
            if texture.type == 0 and not texture.filename.value and texture_id:
                ids.add(texture_id)
        for dependency in sorted(ids):
            game_path = listfile.get(dependency)
            if not game_path:
                raise KeyError("FileDataID %d is absent from the community listfile" % dependency)
            local = source_dir / Path(game_path.replace("\\", "/")).name
            local.write_bytes(reader.open_fdid(dependency))


def walk_objects(root: object) -> Iterable[object]:
    seen = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        yield value
        if isinstance(value, (str, bytes, int, float, bool, type(None), tuple)):
            continue
        if isinstance(value, list):
            stack.extend(value)
            continue
        values = getattr(value, "values", None)
        if isinstance(values, list):
            stack.extend(values)
        attributes = getattr(value, "__dict__", None)
        if attributes:
            stack.extend(attributes.values())


def collect_tracks(root: object, track_base: type) -> List[object]:
    return [value for value in walk_objects(root) if isinstance(value, track_base)]


def assign_sequence_times(root: object) -> None:
    timestamp = 0
    for sequence in root.sequences:
        duration = int(getattr(sequence, "duration", 0) or 0)
        sequence.start_timestamp = timestamp
        sequence.end_timestamp = timestamp + duration
        timestamp = sequence.end_timestamp + 1000


def denormalize_tracks(root: object, modules: Dict[str, object]) -> None:
    common, types, fmt = modules["common"], modules["types"], modules["m2_format"]
    M2Array, M2Range = common.M2Array, fmt.M2Range
    sequences = list(root.sequences)

    def array(value_type: type, values: List[object]) -> object:
        result = M2Array(value_type)
        result.values = list(values)
        result.n_elements = len(result.values)
        return result

    for track in collect_tracks(root, fmt.M2TrackBase):
        timestamp_arrays = track.timestamps.values
        if not timestamp_arrays or not isinstance(timestamp_arrays[0], M2Array):
            continue
        has_values = hasattr(track, "values")
        value_arrays = track.values.values if has_values else []
        value_type = value_arrays[0].type if value_arrays else types.uint32
        flat_times: List[int] = []
        flat_values: List[object] = []
        ranges: List[Tuple[int, int]] = []
        if track.global_sequence >= 0:
            flat_times = [int(value) for value in timestamp_arrays[0].values]
            flat_values = list(value_arrays[0].values) if value_arrays else []
        else:
            last_value = None
            for index, sequence in enumerate(sequences):
                times = timestamp_arrays[index].values if index < len(timestamp_arrays) else []
                values = value_arrays[index].values if index < len(value_arrays) else []
                offset = len(flat_times)
                if times:
                    flat_times.extend(int(value) + sequence.start_timestamp for value in times)
                    if has_values:
                        flat_values.extend(values)
                        last_value = values[-1]
                    ranges.append((offset, offset + len(times) - 1))
                elif has_values and last_value is not None:
                    flat_times.append(sequence.start_timestamp)
                    flat_values.append(last_value)
                    ranges.append((offset, offset))
                else:
                    ranges.append((offset, max(0, offset - 1)))
            ranges.append((0, 0))
        if has_values and flat_values and isinstance(flat_values[0], fmt.M2CompQuaternion):
            converted = []
            for value in flat_values:
                ww, xx, yy, zz = value.to_quaternion()
                converted.append((xx, yy, zz, ww))
            flat_values = converted
            value_type = types.quat
        interpolation = M2Array(M2Range)
        interpolation.values = []
        for minimum, maximum in ranges:
            item = M2Range()
            item.minimum, item.maximum = minimum, maximum
            interpolation.values.append(item)
        interpolation.n_elements = len(interpolation.values)
        track.interpolation_ranges = interpolation
        track.timestamps = array(types.uint32, flat_times)
        if has_values:
            track.values = array(value_type, flat_values)


def stage_sources(shader_id: int, count: int) -> Tuple[int, ...]:
    shader_id &= 0xFFFF
    if count <= 1:
        return (-1,) if shader_id & 0x80 else ((1,) if shader_id & 0x4000 else (0,))
    first = -1 if shader_id & 0x80 else 0
    second = -1 if shader_id & 0x8 else (1 if shader_id & 0x4000 else 0)
    return (first, second)


def find_or_append(table: List[int], values: Tuple[int, ...]) -> int:
    for index in range(len(table) - len(values) + 1):
        if tuple(table[index : index + len(values)]) == values:
            return index
    table.extend(values)
    return len(table) - len(values)


def fix_texture_units(model: object) -> None:
    root = model.root
    coordinates = [0, 1]
    weights = list(root.transparency_lookup_table.values)
    transforms = list(root.texture_transforms_lookup_table.values)
    textures = list(root.texture_lookup_table.values)
    pair_cache = {}
    for skin in model.skins:
        for unit in skin.texture_units.values:
            count = int(unit.texture_count)
            if count not in (1, 2):
                raise ValueError("unsupported textureCount %d" % count)
            unit.texture_coord_combo_index = find_or_append(
                coordinates, stage_sources(unit.shader_id, count)
            )
            for stage in range(count):
                tex = unit.texture_combo_index + stage
                xform = unit.texture_transform_combo_index + stage
                if tex >= len(textures) or textures[tex] >= len(root.textures.values):
                    raise ValueError("texture combo is out of range")
                if xform >= len(transforms) or transforms[xform] >= len(root.texture_transforms.values):
                    raise ValueError("texture transform combo is out of range")
            weight = unit.texture_weight_combo_index
            if weight >= len(weights) or weights[weight] >= len(root.texture_weights.values):
                raise ValueError("texture weight combo is out of range")
            if count == 2:
                track = weights[weight]
                if track not in pair_cache:
                    pair_cache[track] = find_or_append(weights, (track, track))
                unit.texture_weight_combo_index = pair_cache[track]
    root.tex_unit_lookup_table.values = coordinates
    root.tex_unit_lookup_table.n_elements = len(coordinates)
    root.transparency_lookup_table.values = weights
    root.transparency_lookup_table.n_elements = len(weights)


def make_classic(model: object, modules: Dict[str, object]) -> None:
    common, fmt, chunks = modules["common"], modules["m2_format"], modules["m2_chunks"]
    source_root = model.root
    assign_sequence_times(source_root)
    denormalize_tracks(source_root, modules)
    for sequence in source_root.sequences:
        if not hasattr(sequence, "blend_time"):
            sequence.blend_time = max(
                int(getattr(sequence, "blend_time_in", 0)),
                int(getattr(sequence, "blend_time_out", 0)),
            )
    common.M2VersionsManager().set_m2_version(common.M2Versions.CLASSIC)
    root = chunks.MD20()
    for name in list(root.__dict__):
        if name in source_root.__dict__ and name not in {"magic", "version", "m2_version", "_size"}:
            setattr(root, name, getattr(source_root, name))
    model.root = root
    model.version = common.M2Versions.CLASSIC
    for value in walk_objects(root):
        if isinstance(value, fmt.M2TrackBase) and not hasattr(value, "interpolation_ranges"):
            value.interpolation_ranges = common.M2Array(fmt.M2Range)
        if hasattr(value, "m2_version"):
            value.m2_version = common.M2Versions.CLASSIC
        if value.__class__.__name__ == "M2SkinProfile":
            value._size = 44
    for skin in model.skins:
        for value in walk_objects(skin):
            if hasattr(value, "m2_version"):
                value.m2_version = common.M2Versions.CLASSIC
            if value.__class__.__name__ == "M2SkinProfile":
                value._size = 44


def write_classic(model: object, path: Path, modules: Dict[str, object]) -> None:
    skin_class = modules["skin_format"].M2SkinProfile
    bone_class = modules["m2_format"].M2CompBone
    types = modules["types"]
    original_skin_write = skin_class.write
    original_skin_size = getattr(skin_class, "size", None)
    original_bone_write = bone_class.write

    def write_embedded(skin: object, stream: object) -> object:
        position = stream.tell()
        stream.write(b"\0" * 44)
        stream.seek(position)
        skin.vertex_indices.write(stream)
        skin.triangle_indices.write(stream)
        skin.bone_indices.write(stream)
        skin.submeshes.write(stream)
        skin.texture_units.write(stream)
        types.uint32.write(stream, skin.bone_count_max)
        return skin

    def write_classic_bone(bone: object, stream: object) -> object:
        types.int32.write(stream, bone.key_bone_id)
        types.uint32.write(stream, bone.flags)
        types.int16.write(stream, bone.parent_bone)
        types.uint16.write(stream, bone.submesh_id)
        bone.translation.write(stream)
        bone.rotation.write(stream)
        bone.scale.write(stream)
        types.vec3D.write(stream, bone.pivot)
        return bone

    skin_class.write = write_embedded
    skin_class.size = staticmethod(lambda: 44)
    bone_class.write = write_classic_bone
    try:
        model.root.skin_profiles.values = list(model.skins)
        model.root.skin_profiles.n_elements = len(model.skins)
        model.root.num_skin_profiles = len(model.skins)
        with path.open("wb") as stream:
            model.root.write(stream)
            stream.seek(0, 2)
            padding = (16 - stream.tell() % 16) % 16
            stream.write(b"\0" * padding)
    finally:
        skin_class.write = original_skin_write
        if original_skin_size is None:
            delattr(skin_class, "size")
        else:
            skin_class.size = original_skin_size
        bone_class.write = original_bone_write


def convert_all(
    source_root: Path,
    output_root: Path,
    listfile_path: Path,
    modules: Dict[str, object],
) -> List[Dict[str, object]]:
    listfile = read_listfile(listfile_path)
    output_root.mkdir(parents=True, exist_ok=True)
    result = []
    for key, file_id, output_name in MODELS:
        source_dir = source_root / str(file_id)
        source_path = source_dir / output_name
        version = source_version(source_path)
        common = modules["common"]
        common.M2VersionsManager().set_m2_version(
            common.M2Versions.from_expansion_number(expansion_for(version))
        )
        model = modules["m2_file"].M2File(expansion_for(version), filepath=str(source_path))
        dependencies = model.find_model_dependencies()
        skins = []
        for dependency in dependencies.skins:
            name = listfile.get(int(dependency))
            if not name:
                raise KeyError("skin FileDataID %d missing from listfile" % dependency)
            skins.append(str(source_dir / Path(name.replace("\\", "/")).name))
        model.read_additional_files(skins, {})
        textures = []
        for index, texture in enumerate(model.root.textures):
            file_id_value = int(getattr(texture, "fdid", 0) or 0)
            if not file_id_value and model.txid and index < len(model.txid.texture_ids):
                file_id_value = int(model.txid.texture_ids[index])
            if texture.type == 0 and not texture.filename.value:
                game_path = listfile.get(file_id_value)
                if not game_path:
                    raise KeyError("texture FileDataID %d missing from listfile" % file_id_value)
                texture.filename.value = game_path.replace("/", "\\")
                local = source_dir / Path(game_path.replace("\\", "/")).name
                target = output_root.joinpath(*game_path.replace("\\", "/").split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local, target)
                textures.append(str(target.relative_to(output_root)).replace("/", "\\"))
        model.root.global_flags &= 0x17
        if hasattr(model.root, "texture_combiner_combos"):
            model.root.texture_combiner_combos.values = []
            model.root.texture_combiner_combos.n_elements = 0
        for material in model.root.materials:
            if int(material.blending_mode) > 6:
                material.blending_mode = 2
        fix_texture_units(model)
        make_classic(model, modules)
        target = output_root / "Environments" / "Stars" / output_name
        target.parent.mkdir(parents=True, exist_ok=True)
        write_classic(model, target, modules)
        result.append(
            {
                "key": key,
                "source_file_data_id": file_id,
                "source_version": version,
                "output": str(target),
                "output_bytes": target.stat().st_size,
                "skins_embedded": len(model.skins),
                "textures": textures,
            }
        )
    return result
