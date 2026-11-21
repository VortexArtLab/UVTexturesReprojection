import gpu
import numpy as np
from time import perf_counter

from ..constants import TILE_SIZE

_RASTER_SHADER = None

def rasterize(triangles, width, height, settings):
    start = perf_counter()
    tri_data, offsets, indices, tiles_x, tiles_y = _build_tile_bins(
        triangles, width, height
    )
    binning_time = perf_counter() - start

    start = perf_counter()

    tri_buffer = gpu.types.GPUStorageBuf(
        gpu.types.Buffer('FLOAT', (tri_data.size,), tri_data.reshape(-1))
    )
    offsets_buffer = gpu.types.GPUStorageBuf(
        gpu.types.Buffer('INT', (offsets.size,), offsets)
    )
    indices_buffer = gpu.types.GPUStorageBuf(
        gpu.types.Buffer('INT', (indices.size,), indices)
    )

    seed_buffer = gpu.types.GPUStorageBuf(
        gpu.types.Buffer(
            'INT',
            (width * height,),
            np.empty(width * height, dtype=np.int32),
        )
    )
    overlap_buffer = gpu.types.GPUStorageBuf(
        gpu.types.Buffer('INT', (1,), np.zeros(1, dtype=np.int32))
    )

    source_coords = gpu.types.GPUTexture(
        (width, height), format='RG32F'
    )

    upload_time = perf_counter() - start

    shader = _get_raster_shader(gpu)
    shader.bind()

    try:
        shader.storage_block("tri_data", tri_buffer)
        shader.storage_block("tile_offsets", offsets_buffer)
        shader.storage_block("tile_indices", indices_buffer)
        shader.storage_block("seed_map", seed_buffer)
        shader.storage_block("overlap_flag", overlap_buffer)
        shader.image("source_coords_img", source_coords)

        shader.uniform_int("width", width)
        shader.uniform_int("height", height)
        shader.uniform_int("tiles_x", tiles_x)
        shader.uniform_int("tile_size", TILE_SIZE)
        shader.uniform_int(
            "overlap_policy",
            0 if settings.overlap_policy == 'ABORT' else 1,
        )

        start = perf_counter()
        gpu.compute.dispatch(
            shader,
            (width + 15) // 16,
            (height + 15) // 16,
            1,
        )
        dispatch_time = perf_counter() - start

    finally:
        gpu.shader.unbind()

    resources = [
        tri_buffer,
        offsets_buffer,
        indices_buffer,
        seed_buffer,
        overlap_buffer,
    ]

    timings = {
        "binning": binning_time,
        "upload": upload_time,
        "dispatch": dispatch_time,
    }

    return source_coords, seed_buffer, overlap_buffer, resources, timings


def _build_tile_bins(triangles, width, height):
    source_triangles, destination_triangles = triangles[:2]
    triangle_count = len(source_triangles)

    tiles_x = (width + TILE_SIZE - 1) // TILE_SIZE
    tiles_y = (height + TILE_SIZE - 1) // TILE_SIZE
    tile_count = tiles_x * tiles_y

    tri_data = np.empty((triangle_count, 12), dtype=np.float32)
    tri_data[:, :6] = source_triangles.reshape(triangle_count, 6)

    destination_px = destination_triangles * np.array(
        (width, height), dtype=np.float32
    )
    tri_data[:, 6:] = destination_px.reshape(triangle_count, 6)

    a = destination_px[:, 0]
    b = destination_px[:, 1]
    c = destination_px[:, 2]

    area = (
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
        (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )

    min_x = destination_px[:, :, 0].min(axis=1)
    max_x = destination_px[:, :, 0].max(axis=1)
    min_y = destination_px[:, :, 1].min(axis=1)
    max_y = destination_px[:, :, 1].max(axis=1)

    x0 = np.maximum(0, np.ceil(min_x - 0.5).astype(np.int64))
    x1 = np.minimum(width - 1, np.floor(max_x - 0.5).astype(np.int64))
    y0 = np.maximum(0, np.ceil(min_y - 0.5).astype(np.int64))
    y1 = np.minimum(height - 1, np.floor(max_y - 0.5).astype(np.int64))

    valid = (
        (np.abs(area) >= 1e-10) &
        (x0 <= x1) &
        (y0 <= y1)
    )

    tx0 = x0 // TILE_SIZE
    tx1 = x1 // TILE_SIZE
    ty0 = y0 // TILE_SIZE
    ty1 = y1 // TILE_SIZE

    tile_widths = tx1 - tx0 + 1
    tile_heights = ty1 - ty0 + 1

    counts = np.zeros(triangle_count, dtype=np.int64)
    counts[valid] = (
        tile_widths[valid] * tile_heights[valid]
    )

    total_pairs = int(counts.sum())

    if total_pairs == 0:
        raise RuntimeError("No rasterizable triangles")

    if total_pairs > np.iinfo(np.int32).max:
        raise RuntimeError("Tile bin index limit exceeded")

    starts = np.cumsum(counts, dtype=np.int64) - counts

    triangle_ids = np.repeat(
        np.arange(triangle_count, dtype=np.int32),
        counts,
    )

    local = (
        np.arange(total_pairs, dtype=np.int64) -
        np.repeat(starts, counts)
    )

    widths = tile_widths[triangle_ids]

    tile_x = tx0[triangle_ids] + local % widths
    tile_y = ty0[triangle_ids] + local // widths

    tile_ids = (
        tile_y * tiles_x + tile_x
    ).astype(np.int32)

    order = np.argsort(tile_ids, kind='stable')

    sorted_tile_ids = tile_ids[order]
    indices = np.ascontiguousarray(triangle_ids[order])

    tile_counts = np.bincount(
        sorted_tile_ids,
        minlength=tile_count,
    )

    offsets = np.empty(tile_count + 1, dtype=np.int32)
    offsets[0] = 0
    offsets[1:] = np.cumsum(tile_counts, dtype=np.int64)

    return (
        np.ascontiguousarray(tri_data),
        offsets,
        indices,
        tiles_x,
        tiles_y,
    )


def _get_raster_shader(gpu):
    global _RASTER_SHADER
    if _RASTER_SHADER is not None:
        return _RASTER_SHADER

    info = gpu.types.GPUShaderCreateInfo()
    info.storage_buf(0, "float", "tri_data[]", qualifiers={'READ'})
    info.storage_buf(1, "int", "tile_offsets[]", qualifiers={'READ'})
    info.storage_buf(2, "int", "tile_indices[]", qualifiers={'READ'})
    info.storage_buf(3, "int", "seed_map[]", qualifiers={'WRITE'})
    info.storage_buf(4, "int", "overlap_flag[]", qualifiers={'READ', 'WRITE'})
    info.image(0, 'RG32F', "FLOAT_2D", "source_coords_img", qualifiers={'WRITE'})

    for name in ("width", "height", "tiles_x", "tile_size", "overlap_policy"):
        info.push_constant('INT', name)

    info.local_group_size(16, 16)

    info.compute_source("""
    float cross2(vec2 a, vec2 b)
    {
        return a.x * b.y - a.y * b.x;
    }

    bool top_left(vec2 a, vec2 b)
    {
        float dx = b.x - a.x;
        float dy = b.y - a.y;
        return dy > 0.0 || (abs(dy) < 1e-12 && dx < 0.0);
    }

    void main()
    {
        ivec2 p = ivec2(gl_GlobalInvocationID.xy);
        if (p.x >= width || p.y >= height) return;

        int pixel_index = p.y * width + p.x;
        int tile_index = (p.y / tile_size) * tiles_x + p.x / tile_size;

        int begin = tile_offsets[tile_index];
        int end = tile_offsets[tile_index + 1];

        vec2 point = vec2(p) + vec2(0.5);
        vec2 result_uv = vec2(0.0);
        bool found = false;
        const float eps = 1e-7;

        for (int i = begin; i < end; i++) {
            int tri = tile_indices[i];
            int base = tri * 12;

            vec2 s0 = vec2(tri_data[base], tri_data[base + 1]);
            vec2 s1 = vec2(tri_data[base + 2], tri_data[base + 3]);
            vec2 s2 = vec2(tri_data[base + 4], tri_data[base + 5]);

            vec2 a = vec2(tri_data[base + 6], tri_data[base + 7]);
            vec2 b = vec2(tri_data[base + 8], tri_data[base + 9]);
            vec2 c = vec2(tri_data[base + 10], tri_data[base + 11]);

            float area = cross2(b - a, c - a);
            if (abs(area) < 1e-10) continue;

            if (area < 0.0) {
                vec2 temp = b;
                b = c;
                c = temp;

                temp = s1;
                s1 = s2;
                s2 = temp;

                area = -area;
            }

            float e0 = cross2(b - a, point - a);
            float e1 = cross2(c - b, point - b);
            float e2 = cross2(a - c, point - c);

            bool inside =
                (e0 > eps || (abs(e0) <= eps && top_left(a, b))) &&
                (e1 > eps || (abs(e1) <= eps && top_left(b, c))) &&
                (e2 > eps || (abs(e2) <= eps && top_left(c, a)));

            if (!inside) continue;

            if (found && overlap_policy == 0)
                atomicOr(overlap_flag[0], 1);

            found = true;

            float w0 = e1 / area;
            float w1 = e2 / area;
            float w2 = e0 / area;

            result_uv = w0 * s0 + w1 * s1 + w2 * s2;
        }

        if (found) {
            imageStore(source_coords_img, p, vec4(result_uv, 0.0, 0.0));
            seed_map[pixel_index] = pixel_index;
        }
        else {
            imageStore(source_coords_img, p, vec4(0.0));
            seed_map[pixel_index] = -1;
        }
    }
    """)

    _RASTER_SHADER = gpu.shader.create_from_info(info)
    return _RASTER_SHADER

