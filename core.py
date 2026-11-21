import bpy
import numpy as np
from time import perf_counter
from . import geometry

from .models import (
    ReprojectionError,
    UVOverlapError,
    ReprojectionMap,
    TriangleData,
)

from . import materials
from .materials import (
    collect_material_images,
    collect_normal_map_images,
    find_reprojected_image,
    iter_material_node_trees,
    update_materials,
)

from .constants import TILE_SIZE
from . import raster

def ensure_cpu_mapping(mapping, settings):
    start = perf_counter()

    check_gpu_overlap(mapping, settings)

    if mapping.source_coords is None:
        if mapping.gpu_source_coords is None:
            raise ReprojectionError("Source coordinates unavailable")

        data = mapping.gpu_source_coords.read()
        mapping.source_coords = np.frombuffer(
            bytes(data), dtype=np.float32
        ).reshape(mapping.height, mapping.width, 2).copy()

    if mapping.nearest_map is None:
        if mapping.gpu_nearest_map is None:
            raise ReprojectionError("Nearest map unavailable")

        data = mapping.gpu_nearest_map.read()
        mapping.nearest_map = np.frombuffer(
            bytes(data), dtype=np.int32
        ).reshape(mapping.height, mapping.width).copy()

    mapping.gpu_source_coords = None
    mapping.gpu_nearest_map = None
    mapping.gpu_overlap_flag = None
    mapping.gpu_resources.clear()

    return perf_counter() - start


def _find_upstream_images(socket, group_stack, visited):
    images = {}
    context = tuple(node.as_pointer() for node in group_stack)
    key = (socket.as_pointer(), context)

    if key in visited:
        return images

    visited.add(key)

    def add(found):
        images.update(found)

    for link in socket.links:
        node = link.from_node

        if node.type == 'TEX_IMAGE':
            if node.image:
                images[node.image.as_pointer()] = node

        elif node.type == 'REROUTE':
            add(_find_upstream_images(
                node.inputs[0],
                group_stack,
                visited,
            ))

        elif node.type == 'GROUP':
            tree = node.node_tree
            if not tree:
                continue

            outputs = [
                item for item in tree.nodes
                if item.type == 'GROUP_OUTPUT'
            ]
            output_node = next(
                (item for item in outputs if item.is_active_output),
                outputs[0] if outputs else None,
            )

            if output_node is None:
                continue

            output = output_node.inputs.get(link.from_socket.identifier)
            if output is None:
                output = output_node.inputs.get(link.from_socket.name)

            if output is not None:
                add(_find_upstream_images(
                    output,
                    group_stack + (node,),
                    visited,
                ))

        elif node.type == 'GROUP_INPUT':
            if not group_stack:
                continue

            group_instance = group_stack[-1]
            input_socket = group_instance.inputs.get(
                link.from_socket.identifier
            )
            if input_socket is None:
                input_socket = group_instance.inputs.get(
                    link.from_socket.name
                )

            if input_socket is not None:
                add(_find_upstream_images(
                    input_socket,
                    group_stack[:-1],
                    visited,
                ))

        else:
            for input_socket in node.inputs:
                if (
                    input_socket.is_linked
                    and input_socket.type in {'RGBA', 'VECTOR'}
                ):
                    add(_find_upstream_images(
                        input_socket,
                        group_stack,
                        visited,
                    ))

    return images


def validate_settings(obj):
    if obj is None or obj.type != 'MESH':
        return False, "Active object must be a mesh"
    if obj.mode != 'OBJECT':
        return False, "Object must be in Object Mode"

    settings = obj.utr_settings
    uv_names = {uv.name for uv in obj.data.uv_layers}

    if settings.source_uv not in uv_names:
        return False, "Invalid source UV map"
    if settings.destination_uv not in uv_names:
        return False, "Invalid destination UV map"
    if settings.source_uv == settings.destination_uv:
        return False, "Source and destination must be different"
    if not obj.data.polygons:
        return False, "Mesh has no polygons"
    if not settings.textures:
        return False, "Add at least one texture"

    seen = set()

    for item in settings.textures:
        image = item.image
        if image is None:
            return False, "Texture slot has no image"
        if image.as_pointer() in seen:
            return False, f"Duplicate image: {image.name}"
        seen.add(image.as_pointer())

        if image.source not in {'FILE', 'GENERATED'}:
            return False, f"Unsupported image source: {image.name}"
        if image.is_multiview:
            return False, f"Multiview image is unsupported: {image.name}"
        if image.size[0] < 1 or image.size[1] < 1:
            return False, f"Invalid image dimensions: {image.name}"

    return True, ""


def build_uv_triangles(obj):
    settings = obj.utr_settings
    selected_images = [
        item.image for item in settings.textures if item.image
    ]
    normal_maps = collect_normal_map_images(obj, selected_images)
    return geometry.build_uv_triangles(obj, normal_maps)


def _sample(pixels, x, y, mode):
    h, w = pixels.shape[:2]

    if mode == 'NEAREST':
        ix = np.clip(np.floor(x + 0.5).astype(np.int32), 0, w - 1)
        iy = np.clip(np.floor(y + 0.5).astype(np.int32), 0, h - 1)
        return pixels[iy, ix]

    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)
    fx = x - ix
    fy = y - iy

    if mode == 'LINEAR':
        x0 = np.clip(ix, 0, w - 1)
        x1 = np.clip(ix + 1, 0, w - 1)
        y0 = np.clip(iy, 0, h - 1)
        y1 = np.clip(iy + 1, 0, h - 1)

        a = pixels[y0, x0] * (1.0 - fx[:, None]) + pixels[y0, x1] * fx[:, None]
        b = pixels[y1, x0] * (1.0 - fx[:, None]) + pixels[y1, x1] * fx[:, None]
        return a * (1.0 - fy[:, None]) + b * fy[:, None]

    def weights(t):
        t2 = t * t
        t3 = t2 * t
        return np.stack((
            -0.5 * t + t2 - 0.5 * t3,
            1.0 - 2.5 * t2 + 1.5 * t3,
            0.5 * t + 2.0 * t2 - 1.5 * t3,
            -0.5 * t2 + 0.5 * t3,
        ), axis=1)

    wx = weights(fx)
    wy = weights(fy)
    result = np.zeros((len(x), 4), dtype=np.float32)

    for j in range(4):
        sy = np.clip(iy + j - 1, 0, h - 1)
        for i in range(4):
            sx = np.clip(ix + i - 1, 0, w - 1)
            result += pixels[sy, sx] * (wx[:, i] * wy[:, j])[:, None]

    return result


def check_gpu_overlap(mapping, settings):
    if mapping.overlap_checked:
        return

    mapping.overlap_checked = True

    if settings.overlap_policy != 'ABORT':
        mapping.gpu_overlap_flag = None
        return

    flag_buffer = mapping.gpu_overlap_flag
    if flag_buffer is None:
        return

    data = np.frombuffer(bytes(flag_buffer.read()), dtype=np.int32)
    mapping.gpu_overlap_flag = None

    if data[0]:
        raise UVOverlapError("Destination UV overlap detected")


def build_reprojection_map(triangles, width, height, settings):
    from . import gpu_backend

    if width * height > np.iinfo(np.int32).max:
        raise ReprojectionError("Image resolution exceeds supported limits")

    source_triangles = triangles.source_triangles
    destination_triangles = triangles.destination_triangles
    tangent_triangles = triangles.tangent_transforms
    normal_maps = triangles.normal_maps

    has_tangent_maps = tangent_triangles is not None

    if has_tangent_maps and settings.compute_backend == 'GPU':
        raise ReprojectionError(
            "Tangent-space normal maps are temporarily supported "
            "only by the CPU backend"
        )

    use_gpu = settings.compute_backend != 'CPU' and not has_tangent_maps

    gpu_source_coords = None
    gpu_seed_map = None
    gpu_overlap_flag = None
    raster_resources = []
    jfa_resources = []
    gpu_errors = []

    if has_tangent_maps and settings.compute_backend == 'AUTO':
        gpu_errors.append(
            "Tangent-space normal maps require CPU processing"
        )

    raster_backend = "CPU"
    raster_details = {}

    if use_gpu:
        try:
            (
                gpu_source_coords,
                gpu_seed_map,
                gpu_overlap_flag,
                raster_resources,
                raster_details,
            ) = gpu_backend.rasterize(
                triangles, width, height, settings
            )
            raster_backend = "GPU"

        except Exception as exc:
            if settings.compute_backend == 'GPU':
                raise ReprojectionError(
                    f"GPU rasterization failed: {exc}"
                ) from exc
            gpu_errors.append(f"Rasterization: {exc}")

    source_coords = None
    seed_map = None
    tangent_map = None

    if gpu_source_coords is None:
        (
            source_coords,
            seed_map,
            tangent_map,
            raster_details,
        ) = raster.rasterize_cpu(
            triangles,
            width,
            height,
            settings,
        )

    jfa_backend = "CPU"
    nearest_map = None
    gpu_nearest_map = None

    start = perf_counter()

    if gpu_seed_map is not None:
        try:
            gpu_nearest_map, jfa_resources = gpu_backend.jump_flood(
                gpu_seed_map, width, height, gpu_input=True
            )
            jfa_backend = "GPU"

        except Exception as exc:
            if settings.compute_backend == 'GPU':
                raise ReprojectionError(f"GPU JFA failed: {exc}") from exc

            seed_data = np.frombuffer(
                bytes(gpu_seed_map.read()), dtype=np.int32
            ).copy().reshape(height, width)

            nearest_map = _jump_flood(seed_data)
            gpu_nearest_map = None
            raster_resources = (
                [gpu_overlap_flag] if gpu_overlap_flag is not None else []
            )
            gpu_errors.append(f"JFA: {exc}")

    elif use_gpu:
        try:
            gpu_nearest_map, jfa_resources = gpu_backend.jump_flood(
                seed_map, width, height
            )
            jfa_backend = "GPU"

        except Exception as exc:
            if settings.compute_backend == 'GPU':
                raise ReprojectionError(f"GPU JFA failed: {exc}") from exc
            nearest_map = _jump_flood(seed_map)
            gpu_errors.append(f"JFA: {exc}")

    else:
        nearest_map = _jump_flood(seed_map)

    jfa_time = perf_counter() - start

    mapping = ReprojectionMap(
        width,
        height,
        source_coords,
        nearest_map,
        tangent_map,
        normal_maps,
    )

    mapping.gpu_source_coords = gpu_source_coords
    mapping.gpu_nearest_map = gpu_nearest_map
    mapping.gpu_overlap_flag = gpu_overlap_flag
    mapping.gpu_resources = raster_resources + jfa_resources

    if use_gpu:
        start = perf_counter()
        try:
            gpu_backend.create_sampling_maps(mapping)

        except Exception as exc:
            if settings.compute_backend == 'GPU':
                raise ReprojectionError(
                    f"GPU map upload failed: {exc}"
                ) from exc
            gpu_errors.append(f"Map upload: {exc}")

        map_upload_time = perf_counter() - start
    else:
        map_upload_time = 0.0

    mapping.gpu_error = "; ".join(gpu_errors)

    raster_total = sum(raster_details.values())
    timings = {
        "raster": raster_total,
        "raster_backend": raster_backend,
        "raster_details": raster_details,
        "jfa": jfa_time,
        "jfa_backend": jfa_backend,
        "jfa_async": jfa_backend == "GPU",
        "map_upload": map_upload_time,
        "backend": jfa_backend,
    }

    if gpu_errors:
        timings["fallback_reason"] = "; ".join(gpu_errors)

    return mapping, timings


def _jump_flood_pass(current, target, w, h, step):
    for y0 in range(0, h, TILE_SIZE):
        y1 = min(y0 + TILE_SIZE, h)

        for x0 in range(0, w, TILE_SIZE):
            x1 = min(x0 + TILE_SIZE, w)

            best = current[y0:y1, x0:x1].copy()
            ys = np.arange(y0, y1, dtype=np.int64)[:, None]
            xs = np.arange(x0, x1, dtype=np.int64)[None, :]

            valid = best >= 0
            seed_x = np.where(valid, best % w, 0).astype(np.int64)
            seed_y = np.where(valid, best // w, 0).astype(np.int64)

            dx = seed_x - xs
            dy = seed_y - ys
            best_dist = dx * dx + dy * dy
            best_dist[~valid] = np.iinfo(np.int64).max

            for oy in (-step, 0, step):
                ay = max(y0, -oy)
                by = min(y1, h - oy)

                if ay >= by:
                    continue

                for ox in (-step, 0, step):
                    if ox == 0 and oy == 0:
                        continue

                    ax = max(x0, -ox)
                    bx = min(x1, w - ox)

                    if ax >= bx:
                        continue

                    candidates = current[
                        ay + oy:by + oy,
                        ax + ox:bx + ox,
                    ]

                    valid = candidates >= 0
                    if not np.any(valid):
                        continue

                    cx = np.where(valid, candidates % w, 0).astype(np.int64)
                    cy = np.where(valid, candidates // w, 0).astype(np.int64)

                    dx = cx - np.arange(ax, bx, dtype=np.int64)[None, :]
                    dy = cy - np.arange(ay, by, dtype=np.int64)[:, None]
                    dist = dx * dx + dy * dy

                    sy = slice(ay - y0, by - y0)
                    sx = slice(ax - x0, bx - x0)

                    best_sub = best[sy, sx]
                    dist_sub = best_dist[sy, sx]

                    update = valid & (dist < dist_sub)

                    best_sub[update] = candidates[update]
                    dist_sub[update] = dist[update]

            target[y0:y1, x0:x1] = best


def _jump_flood(seed_map):
    h, w = seed_map.shape
    current = seed_map
    buffer = np.empty_like(current)

    step = 1 << max(0, (max(w, h) - 1).bit_length() - 1)

    while step:
        _jump_flood_pass(current, buffer, w, h, step)
        current, buffer = buffer, current
        step //= 2

    _jump_flood_pass(current, buffer, w, h, 1)
    current, buffer = buffer, current

    return current


def _reproject_image_cpu(image, mapping, settings):
    timings = {}
    w, h = mapping.width, mapping.height
    normal_info = mapping.normal_maps.get(image.as_pointer())
    tangent_normal = bool(
        normal_info and normal_info["space"] == 'TANGENT'
    )

    if tangent_normal and mapping.tangent_map is None:
        raise ReprojectionError(
            f"Tangent map unavailable for image: {image.name}"
        )

    if tuple(image.size) != (w, h):
        raise ReprojectionError(f"Resolution mismatch: {image.name}")

    try:
        start = perf_counter()
        if image.source == 'FILE' and not image.has_data:
            image.reload()

        source_pixels = np.empty((h, w, 4), dtype=np.float32)
        image.pixels.foreach_get(source_pixels.ravel())
        timings["read"] = perf_counter() - start

        start = perf_counter()
        output = np.zeros((h, w, 4), dtype=np.float32)
        source_flat = mapping.source_coords.reshape(-1, 2)
        tangent_flat = (
            mapping.tangent_map.reshape(-1, 4)
            if tangent_normal else None
        )
        normal_transform_time = 0.0
        radius_sq = settings.padding ** 2

        for y0 in range(0, h, TILE_SIZE):
            y1 = min(y0 + TILE_SIZE, h)

            for x0 in range(0, w, TILE_SIZE):
                x1 = min(x0 + TILE_SIZE, w)
                ids = mapping.nearest_map[y0:y1, x0:x1]
                valid = ids >= 0

                if settings.dilation_mode == 'FIXED':
                    safe_ids = np.maximum(ids, 0)
                    ys = np.arange(y0, y1, dtype=np.int64)[:, None]
                    xs = np.arange(x0, x1, dtype=np.int64)[None, :]
                    dx = safe_ids % w - xs
                    dy = safe_ids // w - ys
                    valid &= (dx * dx + dy * dy) <= radius_sq

                if not np.any(valid):
                    continue

                coords = source_flat[ids[valid]]
                sampled = _sample(
                    source_pixels,
                    coords[:, 0] * w - 0.5,
                    coords[:, 1] * h - 0.5,
                    settings.interpolation,
                ).copy()

                if tangent_normal:
                    start_normal = perf_counter()

                    matrix = tangent_flat[ids[valid]]
                    normal = sampled[:, :3] * 2.0 - 1.0

                    directx = normal_info["convention"] == 'DIRECTX'

                    if directx:
                        normal[:, 1] *= -1.0

                    x = matrix[:, 0] * normal[:, 0] + matrix[:, 1] * normal[:, 1]
                    y = matrix[:, 2] * normal[:, 0] + matrix[:, 3] * normal[:, 1]

                    normal[:, 0] = x
                    normal[:, 1] = y

                    normal_length = np.linalg.norm(normal, axis=1)
                    normal /= np.maximum(normal_length[:, None], 1e-8)

                    if directx:
                        normal[:, 1] *= -1.0

                    sampled[:, :3] = normal * 0.5 + 0.5

                    normal_transform_time += perf_counter() - start_normal

                output[y0:y1, x0:x1][valid] = sampled

        timings["sample"] = perf_counter() - start

        if tangent_normal:
            timings["normal_transform"] = normal_transform_time

        start = perf_counter()
        result = bpy.data.images.new(
            name=f"{image.name}{settings.output_suffix}",
            width=w,
            height=h,
            alpha=True,
            float_buffer=image.is_float,
            is_data=image.colorspace_settings.is_data,
        )
        timings["create"] = perf_counter() - start

        try:
            result.colorspace_settings.name = image.colorspace_settings.name
            result.alpha_mode = image.alpha_mode
            result.display_aspect = image.display_aspect
            result.file_format = image.file_format

            start = perf_counter()
            result.pixels.foreach_set(output.ravel())
            result.update()
            timings["write"] = perf_counter() - start

        except Exception:
            bpy.data.images.remove(result)
            raise

        result["utr_source_image"] = image.name
        result["utr_output_suffix"] = settings.output_suffix
        return result, timings

    except Exception as exc:
        if isinstance(exc, ReprojectionError):
            raise
        raise ReprojectionError(f"Cannot process {image.name}: {exc}") from exc


def reproject_image(image, mapping, settings):
    map_readback = 0.0

    if settings.compute_backend != 'CPU':
        gpu_ready = (
            mapping.gpu_source_coords is not None
            and mapping.gpu_nearest_map is not None
        )

        if gpu_ready:
            try:
                from . import gpu_backend
                result, timings = gpu_backend.reproject_image(
                    image, mapping, settings
                )
                timings["backend"] = "GPU"
                return result, timings

            except UVOverlapError:
                raise

            except Exception as exc:
                if settings.compute_backend == 'GPU':
                    raise ReprojectionError(
                        f"GPU sampling failed: {exc}"
                    ) from exc

                mapping.gpu_error = "; ".join(filter(
                    None, (mapping.gpu_error, f"Sampling: {exc}")
                ))

        elif settings.compute_backend == 'GPU':
            raise ReprojectionError(
                f"GPU sampling unavailable: "
                f"{mapping.gpu_error or 'GPU maps missing'}"
            )

        map_readback = ensure_cpu_mapping(mapping, settings)

    result, timings = _reproject_image_cpu(image, mapping, settings)
    timings["map_readback"] = map_readback

    if settings.compute_backend == 'AUTO' and mapping.gpu_error:
        timings["backend"] = "CPU fallback"
        timings["fallback_reason"] = mapping.gpu_error
    else:
        timings["backend"] = "CPU"

    return result, timings
