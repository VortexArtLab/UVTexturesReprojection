import numpy as np
from time import perf_counter

from . import geometry, jfa, materials, raster
from .constants import TILE_SIZE
from .models import ReprojectionError, ReprojectionMap, UVOverlapError


def build_uv_triangles(obj):
    settings = obj.utr_settings
    selected_images = [
        item.image for item in settings.textures if item.image
    ]
    normal_maps = materials.collect_normal_map_images(obj, selected_images)
    return geometry.build_uv_triangles(obj, normal_maps)


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

            nearest_map = jfa.jump_flood(seed_data)
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
            nearest_map = jfa.jump_flood(seed_map)
            gpu_errors.append(f"JFA: {exc}")

    else:
        nearest_map = jfa.jump_flood(seed_map)

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


def build_uv_triangles(obj):
    settings = obj.utr_settings
    selected_images = [
        item.image for item in settings.textures if item.image
    ]
    normal_maps = materials.collect_normal_map_images(obj, selected_images)
    return geometry.build_uv_triangles(obj, normal_maps)


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