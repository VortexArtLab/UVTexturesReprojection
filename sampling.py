import bpy
import numpy as np
from time import perf_counter
from .constants import TILE_SIZE
from .models import ReprojectionError

MAX_SAMPLING_PLAN_BYTES = 1792 * 1024 * 1024
MAX_SAMPLING_PLAN_MAX_DIMENSION = 8192


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


def _build_sampling_plan(mapping, settings):
    w, h = mapping.width, mapping.height
    if max(w, h) >= 16384:
        return None
    index_dtype = np.dtype(
        np.uint16 if max(w, h) <= 65536 else np.uint32
    )
    tile_specs = []
    valid_count = 0
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

            count = int(np.count_nonzero(valid))
            valid_count += count

            weight_dtype = np.dtype(
                np.result_type(mapping.source_coords.dtype, np.int32)
            )

            plan_bytes = w * h + valid_count * (
                4 * index_dtype.itemsize
                + 2 * weight_dtype.itemsize
            )

            if plan_bytes > MAX_SAMPLING_PLAN_BYTES:
                return None

            tile_specs.append((y0, y1, x0, x1, valid, count))

    source_flat = mapping.source_coords.reshape(-1, 2)
    tiles = []

    for y0, y1, x0, x1, valid, count in tile_specs:
        if count:
            ids = mapping.nearest_map[y0:y1, x0:x1]
            coords = source_flat[ids[valid]]

            sx = coords[:, 0] * w - 0.5
            sy = coords[:, 1] * h - 0.5
            ix = np.floor(sx).astype(np.int32)
            iy = np.floor(sy).astype(np.int32)
            fx = sx - ix
            fy = sy - iy

            sample_x0 = np.clip(ix, 0, w - 1).astype(index_dtype)
            sample_x1 = np.clip(ix + 1, 0, w - 1).astype(index_dtype)
            sample_y0 = np.clip(iy, 0, h - 1).astype(index_dtype)
            sample_y1 = np.clip(iy + 1, 0, h - 1).astype(index_dtype)

        else:
            sample_x0 = np.empty(0, dtype=index_dtype)
            sample_x1 = np.empty(0, dtype=index_dtype)
            sample_y0 = np.empty(0, dtype=index_dtype)
            sample_y1 = np.empty(0, dtype=index_dtype)
            fx = np.empty(0, dtype=np.float32)
            fy = np.empty(0, dtype=np.float32)

        tiles.append((
            y0, y1, x0, x1,
            valid,
            sample_x0, sample_x1,
            sample_y0, sample_y1,
            fx, fy, count,
        ))

    return {"tiles": tiles, "bytes": plan_bytes}


def _sample_linear_plan(pixels, x0, x1, y0, y1, fx, fy):
    a = (
        pixels[y0, x0] * (1.0 - fx[:, None])
        + pixels[y0, x1] * fx[:, None]
    )
    b = (
        pixels[y1, x0] * (1.0 - fx[:, None])
        + pixels[y1, x1] * fx[:, None]
    )
    return a * (1.0 - fy[:, None]) + b * fy[:, None]



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
        plan_build_time = 0.0
        plan = None
        plan_status = "not_applicable"

        if settings.interpolation == 'LINEAR':
            if mapping.sampling_plan is None:
                start = perf_counter()
                mapping.sampling_plan = _build_sampling_plan(
                    mapping, settings
                )
                if mapping.sampling_plan is None:
                    mapping.sampling_plan = False
                plan_build_time = perf_counter() - start

            if mapping.sampling_plan is False:
                plan_status = "memory_limit"
            else:
                plan = mapping.sampling_plan
                plan_status = "cached"

        start = perf_counter()
        if image.source == 'FILE' and not image.has_data:
            image.reload()

        source_pixels = np.empty((h, w, 4), dtype=np.float32)
        image.pixels.foreach_get(source_pixels.ravel())
        timings["read"] = perf_counter() - start

        start = perf_counter()
        output = np.zeros((h, w, 4), dtype=np.float32)
        source_flat = (
            None if plan is not None
            else mapping.source_coords.reshape(-1, 2)
        )
        tangent_flat = (
            mapping.tangent_map.reshape(-1, 4)
            if tangent_normal else None
        )
        normal_transform_time = 0.0
        radius_sq = settings.padding ** 2
        plan_iter = iter(plan["tiles"]) if plan is not None else None

        for y0 in range(0, h, TILE_SIZE):
            y1 = min(y0 + TILE_SIZE, h)

            for x0 in range(0, w, TILE_SIZE):
                x1 = min(x0 + TILE_SIZE, w)
                ids = mapping.nearest_map[y0:y1, x0:x1]

                if plan_iter is not None:
                    (
                        _, _, _, _, valid,
                        sample_x0, sample_x1,
                        sample_y0, sample_y1,
                        fx, fy, count,
                    ) = next(plan_iter)

                    if count == 0:
                        continue

                else:
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

                if plan_iter is not None:
                    sampled = _sample_linear_plan(
                        source_pixels,
                        sample_x0,
                        sample_x1,
                        sample_y0,
                        sample_y1,
                        fx,
                        fy,
                    )
                else:
                    coords = source_flat[ids[valid]]
                    sampled = _sample(
                        source_pixels,
                        coords[:, 0] * w - 0.5,
                        coords[:, 1] * h - 0.5,
                        settings.interpolation,
                    )

                if tangent_normal:
                    start_normal = perf_counter()
                    matrix = tangent_flat[ids[valid]]
                    normal = sampled[:, :3] * 2.0 - 1.0
                    directx = normal_info["convention"] == 'DIRECTX'

                    if directx:
                        normal[:, 1] *= -1.0

                    x = (
                        matrix[:, 0] * normal[:, 0]
                        + matrix[:, 1] * normal[:, 1]
                    )
                    y = (
                        matrix[:, 2] * normal[:, 0]
                        + matrix[:, 3] * normal[:, 1]
                    )

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

        if plan_build_time:
            timings["plan_build"] = plan_build_time

        timings["sampling_plan"] = plan_status

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
        raise ReprojectionError(
            f"Cannot process {image.name}: {exc}"
        ) from exc
