
import bpy
import numpy as np

TILE_SIZE = 256

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


class ReprojectionError(Exception):
    pass


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

    source = mesh.uv_layers[settings.source_uv].data
    destination = mesh.uv_layers[settings.destination_uv].data
    triangles = []

    for triangle in mesh.loop_triangles:
        loops = triangle.loops
        src = np.array([source[i].uv[:] for i in loops], dtype=np.float32)
        dst = np.array([destination[i].uv[:] for i in loops], dtype=np.float32)

        for uv in (src, dst):
            if not np.all(np.isfinite(uv)):
                raise ReprojectionError("UV coordinates contain invalid values")
            if np.any(uv < -1e-6) or np.any(uv > 1.000001):
                raise ReprojectionError("UV coordinates outside 0-1 are unsupported")

        triangles.append((np.clip(src, 0.0, 1.0), np.clip(dst, 0.0, 1.0)))

    if not triangles:
        raise ReprojectionError("No triangles found")

    return triangles


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



def _rasterize_triangle(source_pixels, output, seed_map, src, dst, settings):
    h, w = output.shape[:2]
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
                raise ReprojectionError("Destination UV overlap detected")

            w0 = e1[inside] / area
            w1 = e2[inside] / area
            w2 = e0[inside] / area

            uv = (
                w0[:, None] * src[0] +
                w1[:, None] * src[1] +
                w2[:, None] * src[2]
            )

            ix = xx[inside]
            iy = yy[inside]

            sampled = _sample(
                source_pixels,
                uv[:, 0] * w - 0.5,
                uv[:, 1] * h - 0.5,
                settings.interpolation,
            )

            output[ty:ty1, tx:tx1][inside] = sampled
            seeds[inside] = (iy.astype(np.int64) * w + ix).astype(np.int32)



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


def _dilate_pixels(pixels, seed_map, settings):
    h, w = seed_map.shape

    if settings.dilation_mode == 'FIXED' and settings.padding <= 0:
        return

    nearest = _jump_flood(seed_map)
    flat = pixels.reshape(-1, 4)
    radius_sq = settings.padding ** 2

    for y0 in range(0, h, TILE_SIZE):
        y1 = min(y0 + TILE_SIZE, h)

        for x0 in range(0, w, TILE_SIZE):
            x1 = min(x0 + TILE_SIZE, w)

            ids = nearest[y0:y1, x0:x1]

            ys = np.arange(y0, y1, dtype=np.int64)[:, None]
            xs = np.arange(x0, x1, dtype=np.int64)[None, :]
            own = ys * w + xs

            valid = (ids >= 0) & (ids != own)

            if settings.dilation_mode == 'FIXED':
                sx = np.where(ids >= 0, ids % w, 0).astype(np.int64)
                sy = np.where(ids >= 0, ids // w, 0).astype(np.int64)

                dx = sx - xs
                dy = sy - ys

                valid &= dx * dx + dy * dy <= radius_sq

            if np.any(valid):
                tile = pixels[y0:y1, x0:x1]
                tile[valid] = flat[ids[valid]]


def reproject_image(image, triangles, settings):
    try:
        if image.source == 'FILE' and not image.has_data:
            image.reload()

        w, h = map(int, image.size)
        if w * h > np.iinfo(np.int32).max:
            raise ReprojectionError("Image resolution exceeds supported limits")

        source_pixels = np.empty((h, w, 4), dtype=np.float32)
        image.pixels.foreach_get(source_pixels.ravel())

    except Exception as exc:
        raise ReprojectionError(f"Cannot read {image.name}: {exc}") from exc

    output = np.zeros((h, w, 4), dtype=np.float32)
    seed_map = np.full((h, w), -1, dtype=np.int32)

    for src, dst in triangles:
        _rasterize_triangle(
            source_pixels, output, seed_map, src, dst, settings
        )

    if seed_map.max() < 0:
        raise ReprojectionError(f"No pixels generated for {image.name}")

    _dilate_pixels(output, seed_map, settings)

    result = bpy.data.images.new(
        name=f"{image.name}{settings.output_suffix}",
        width=w,
        height=h,
        alpha=True,
        float_buffer=image.is_float,
        is_data=image.colorspace_settings.is_data,
    )

    try:
        result.colorspace_settings.name = image.colorspace_settings.name
        result.alpha_mode = image.alpha_mode
        result.display_aspect = image.display_aspect
        result.file_format = image.file_format
        result.pixels.foreach_set(output.ravel())
        result.update()
    except Exception:
        bpy.data.images.remove(result)
        raise

    return result
