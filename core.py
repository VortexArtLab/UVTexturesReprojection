
import bpy
import numpy as np
from time import perf_counter

TILE_SIZE = 256


class ReprojectionError(Exception):
    pass


class UVOverlapError(ReprojectionError):
    pass


class ReprojectionMap:
    __slots__ = (
        "width", "height",
        "source_coords", "nearest_map",
        "gpu_source_coords", "gpu_nearest_map",
        "gpu_overlap_flag", "gpu_resources",
        "gpu_error", "overlap_checked",
    )

    def __init__(self, width, height, source_coords, nearest_map):
        self.width = width
        self.height = height
        self.source_coords = source_coords
        self.nearest_map = nearest_map
        self.gpu_source_coords = None
        self.gpu_nearest_map = None
        self.gpu_overlap_flag = None
        self.gpu_resources = []
        self.gpu_error = ""
        self.overlap_checked = False


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


def collect_material_images(obj):
    images = []
    visited = set()
    found = set()

    def visit(tree):
        if not tree:
            return

        pointer = tree.as_pointer()
        if pointer in visited:
            return
        visited.add(pointer)

        for node in tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                image = node.image
                pointer = image.as_pointer()
                if pointer not in found:
                    found.add(pointer)
                    images.append(image)
            elif node.type == 'GROUP':
                visit(node.node_tree)

    for slot in obj.material_slots:
        material = slot.material
        if material and material.use_nodes:
            visit(material.node_tree)

    return images


def iter_material_node_trees(obj):
    visited = set()
    stack = [
        slot.material.node_tree
        for slot in obj.material_slots
        if slot.material and slot.material.use_nodes and slot.material.node_tree
    ]

    while stack:
        tree = stack.pop()
        pointer = tree.as_pointer()

        if pointer in visited:
            continue

        visited.add(pointer)
        yield tree

        for node in tree.nodes:
            if node.type == 'GROUP' and node.node_tree:
                stack.append(node.node_tree)


def find_reprojected_image(image, suffix):
    matches = [
        candidate for candidate in bpy.data.images
        if candidate.get("utr_source_image") == image.name
        and candidate.get("utr_output_suffix") == suffix
    ]

    if matches:
        return matches[-1]

    return bpy.data.images.get(f"{image.name}{suffix}")


def update_materials(obj, settings):
    uv_names = {uv.name for uv in obj.data.uv_layers}

    if settings.destination_uv not in uv_names:
        raise ReprojectionError("Invalid destination UV map")

    replacements = {}
    missing = []

    for item in settings.textures:
        source = item.image
        if source is None:
            continue

        output = find_reprojected_image(source, settings.output_suffix)

        if output is None:
            missing.append(source.name)
            continue

        replacements[source.as_pointer()] = output
        replacements[output.as_pointer()] = output

    if not replacements:
        return 0, missing

    updated_nodes = 0

    for tree in iter_material_node_trees(obj):
        targets = []

        for node in tree.nodes:
            if node.type != 'TEX_IMAGE' or node.image is None:
                continue

            output = replacements.get(node.image.as_pointer())
            if output is None:
                continue

            node.image = output
            targets.append(node)

        if not targets:
            continue

        target_sockets = {
            node.inputs["Vector"].as_pointer()
            for node in targets
            if node.inputs.get("Vector") is not None
        }

        existing_uv = {}

        for node in targets:
            vector = node.inputs.get("Vector")
            if vector is None:
                continue

            for link in vector.links:
                source = link.from_node
                if source.bl_idname == 'ShaderNodeUVMap':
                    existing_uv[source.as_pointer()] = source

        reusable_uv = []

        for uv_node in existing_uv.values():
            output = uv_node.outputs.get("UV")
            if output is None:
                continue

            if all(
                link.to_socket.as_pointer() in target_sockets
                for link in output.links
            ):
                reusable_uv.append(uv_node)

        if reusable_uv:
            for uv_node in reusable_uv:
                uv_node.uv_map = settings.destination_uv
            shared_uv = reusable_uv[0]
        else:
            shared_uv = tree.nodes.new("ShaderNodeUVMap")
            shared_uv.uv_map = settings.destination_uv
            shared_uv.label = "Destination UV"

            texture_x = min(node.location.x for node in targets)
            texture_y = sum(node.location.y for node in targets) / len(targets)

            shared_uv.location = (texture_x - 220, texture_y)
            reusable_uv = [shared_uv]

        for node in targets:
            vector = node.inputs.get("Vector")
            if vector is None:
                continue

            if any(link.from_node in reusable_uv for link in vector.links):
                continue

            for link in list(vector.links):
                tree.links.remove(link)

            tree.links.new(shared_uv.outputs["UV"], vector)

        updated_nodes += len(targets)

    return updated_nodes, missing





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
    mesh = obj.data
    settings = obj.utr_settings

    mesh.calc_loop_triangles()
    triangle_count = len(mesh.loop_triangles)

    if not triangle_count:
        raise ReprojectionError("No triangles found")

    source_layer = mesh.uv_layers.get(settings.source_uv)
    destination_layer = mesh.uv_layers.get(settings.destination_uv)

    if source_layer is None or destination_layer is None:
        raise ReprojectionError("Source or destination UV map not found")

    loop_indices = np.empty(triangle_count * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", loop_indices)
    loop_indices = loop_indices.reshape(-1, 3)

    loop_count = len(mesh.loops)
    source_uv = np.empty(loop_count * 2, dtype=np.float32)
    destination_uv = np.empty(loop_count * 2, dtype=np.float32)

    source_layer.data.foreach_get("uv", source_uv)
    destination_layer.data.foreach_get("uv", destination_uv)

    source_uv = source_uv.reshape(-1, 2)
    destination_uv = destination_uv.reshape(-1, 2)

    source_triangles = np.ascontiguousarray(source_uv[loop_indices])
    destination_triangles = np.ascontiguousarray(destination_uv[loop_indices])

    for name, uv in (
        ("Source", source_triangles),
        ("Destination", destination_triangles),
    ):
        if not np.all(np.isfinite(uv)):
            raise ReprojectionError(f"{name} UV coordinates contain invalid values")

        if np.any(uv < -1e-6) or np.any(uv > 1.000001):
            raise ReprojectionError(f"{name} UV coordinates outside 0-1 are unsupported")

        np.clip(uv, 0.0, 1.0, out=uv)

    return source_triangles, destination_triangles


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


def _top_left(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return dy > 0.0 or (abs(dy) < 1e-12 and dx < 0.0)


def _rasterize_triangle(source_coords, seed_map, src, dst, settings):
    h, w = seed_map.shape
    dst = dst * np.array((w, h), dtype=np.float32)

    area = (
        (dst[1, 0] - dst[0, 0]) * (dst[2, 1] - dst[0, 1]) -
        (dst[1, 1] - dst[0, 1]) * (dst[2, 0] - dst[0, 0])
    )
    if abs(area) < 1e-10:
        return

    if area < 0.0:
        dst = dst[[0, 2, 1]]
        src = src[[0, 2, 1]]
        area = -area

    x0 = max(0, int(np.ceil(dst[:, 0].min() - 0.5)))
    x1 = min(w - 1, int(np.floor(dst[:, 0].max() - 0.5)))
    y0 = max(0, int(np.ceil(dst[:, 1].min() - 0.5)))
    y1 = min(h - 1, int(np.floor(dst[:, 1].max() - 0.5)))

    if x0 > x1 or y0 > y1:
        return

    a, b, c = dst

    for ty in range(y0, y1 + 1, TILE_SIZE):
        ty1 = min(ty + TILE_SIZE, y1 + 1)

        for tx in range(x0, x1 + 1, TILE_SIZE):
            tx1 = min(tx + TILE_SIZE, x1 + 1)

            xx, yy = np.meshgrid(
                np.arange(tx, tx1, dtype=np.int32),
                np.arange(ty, ty1, dtype=np.int32),
            )

            px = xx.astype(np.float32) + 0.5
            py = yy.astype(np.float32) + 0.5

            e0 = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
            e1 = (c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0])
            e2 = (a[0] - c[0]) * (py - c[1]) - (a[1] - c[1]) * (px - c[0])

            eps = 1e-7
            inside = (
                ((e0 > eps) | ((np.abs(e0) <= eps) & _top_left(a, b))) &
                ((e1 > eps) | ((np.abs(e1) <= eps) & _top_left(b, c))) &
                ((e2 > eps) | ((np.abs(e2) <= eps) & _top_left(c, a)))
            )

            if not np.any(inside):
                continue

            seeds = seed_map[ty:ty1, tx:tx1]

            if settings.overlap_policy == 'ABORT' and np.any(inside & (seeds >= 0)):
                raise UVOverlapError("Destination UV overlap detected")

            w0 = e1[inside] / area
            w1 = e2[inside] / area
            w2 = e0[inside] / area

            uv = (
                w0[:, None] * src[0] +
                w1[:, None] * src[1] +
                w2[:, None] * src[2]
            )

            source_coords[ty:ty1, tx:tx1][inside] = uv
            seeds[inside] = (yy[inside] * w + xx[inside])


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

    gpu_source_coords = None
    gpu_seed_map = None
    gpu_overlap_flag = None
    raster_resources = []
    jfa_resources = []
    gpu_errors = []

    raster_backend = "CPU"
    raster_details = {}

    if settings.compute_backend != 'CPU':
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
    raster_time = 0.0

    if gpu_source_coords is None:
        source_coords = np.zeros(
            (height, width, 2), dtype=np.float32
        )
        seed_map = np.full(
            (height, width), -1, dtype=np.int32
        )

        start = perf_counter()
        source_triangles, destination_triangles = triangles

        for index in range(len(source_triangles)):
            _rasterize_triangle(
                source_coords,
                seed_map,
                source_triangles[index],
                destination_triangles[index],
                settings,
            )
        raster_time = perf_counter() - start

        if not np.any(seed_map >= 0):
            raise ReprojectionError("No pixels generated")

        raster_details = {"cpu": raster_time}

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
            raster_resources = ([gpu_overlap_flag] if gpu_overlap_flag is not None else [])
            gpu_errors.append(f"JFA: {exc}")

    elif settings.compute_backend != 'CPU':
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
        width, height, source_coords, nearest_map
    )

    mapping.gpu_source_coords = gpu_source_coords
    mapping.gpu_nearest_map = gpu_nearest_map
    mapping.gpu_overlap_flag = gpu_overlap_flag
    mapping.gpu_resources = raster_resources + jfa_resources

    if settings.compute_backend != 'CPU':
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
                )
                output[y0:y1, x0:x1][valid] = sampled

        timings["sample"] = perf_counter() - start

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
