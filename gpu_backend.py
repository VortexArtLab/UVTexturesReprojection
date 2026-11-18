import bpy
import gpu
import numpy as np
from time import perf_counter

_SHADER = None
_SAMPLE_SHADER = None
_RASTER_SHADER = None
TILE_SIZE = 256


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
    source_triangles, destination_triangles = triangles
    triangle_count = len(source_triangles)

    tiles_x = (width + TILE_SIZE - 1) // TILE_SIZE
    tiles_y = (height + TILE_SIZE - 1) // TILE_SIZE
    bins = [[] for _ in range(tiles_x * tiles_y)]

    tri_data = np.empty((triangle_count, 12), dtype=np.float32)
    tri_data[:, :6] = source_triangles.reshape(triangle_count, 6)

    destination_px = destination_triangles * np.array(
        (width, height), dtype=np.float32
    )
    tri_data[:, 6:] = destination_px.reshape(triangle_count, 6)

    for index in range(triangle_count):
        dst = destination_px[index]
        a, b, c = dst

        area = (
            (b[0] - a[0]) * (c[1] - a[1]) -
            (b[1] - a[1]) * (c[0] - a[0])
        )

        if abs(area) < 1e-10:
            continue

        x0 = max(0, int(np.ceil(dst[:, 0].min() - 0.5)))
        x1 = min(width - 1, int(np.floor(dst[:, 0].max() - 0.5)))
        y0 = max(0, int(np.ceil(dst[:, 1].min() - 0.5)))
        y1 = min(height - 1, int(np.floor(dst[:, 1].max() - 0.5)))

        if x0 > x1 or y0 > y1:
            continue

        for ty in range(y0 // TILE_SIZE, y1 // TILE_SIZE + 1):
            for tx in range(x0 // TILE_SIZE, x1 // TILE_SIZE + 1):
                bins[ty * tiles_x + tx].append(index)

    offsets = np.zeros(len(bins) + 1, dtype=np.int32)
    indices = []

    for i, bucket in enumerate(bins):
        offsets[i] = len(indices)
        indices.extend(bucket)

    offsets[-1] = len(indices)

    if not indices:
        raise RuntimeError("No rasterizable triangles")

    return (
        np.ascontiguousarray(tri_data),
        offsets,
        np.asarray(indices, dtype=np.int32),
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


def _get_sample_shader(gpu):
    global _SAMPLE_SHADER
    if _SAMPLE_SHADER is not None:
        return _SAMPLE_SHADER

    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, 'FLOAT_2D', "source_tex")
    info.sampler(1, 'FLOAT_2D', "coords_tex")
    info.storage_buf(0, "int", "nearest_map[]", qualifiers={'READ'})
    info.image(0, 'RGBA32F', "FLOAT_2D", "output_img", qualifiers={'WRITE'})
    info.push_constant('INT', "width")
    info.push_constant('INT', "height")
    info.push_constant('INT', "interpolation")
    info.push_constant('INT', "fixed_padding")
    info.push_constant('INT', "padding")
    info.local_group_size(16, 16)

    info.compute_source("""
    vec4 cubic_weights(float t)
    {
        float t2 = t * t;
        float t3 = t2 * t;
        return vec4(
            -0.5 * t + t2 - 0.5 * t3,
            1.0 - 2.5 * t2 + 1.5 * t3,
            0.5 * t + 2.0 * t2 - 1.5 * t3,
            -0.5 * t2 + 0.5 * t3
        );
    }

    ivec2 clamp_pixel(ivec2 p)
    {
        return clamp(p, ivec2(0), ivec2(width - 1, height - 1));
    }

    vec4 sample_source(vec2 uv)
    {
        vec2 pos = uv * vec2(width, height) - 0.5;
        ivec2 base = ivec2(floor(pos));
        vec2 f = fract(pos);

        if (interpolation == 0)
            return texelFetch(source_tex, clamp_pixel(ivec2(floor(pos + 0.5))), 0);

        if (interpolation == 1) {
            vec4 a = texelFetch(source_tex, clamp_pixel(base), 0);
            vec4 b = texelFetch(source_tex, clamp_pixel(base + ivec2(1, 0)), 0);
            vec4 c = texelFetch(source_tex, clamp_pixel(base + ivec2(0, 1)), 0);
            vec4 d = texelFetch(source_tex, clamp_pixel(base + ivec2(1, 1)), 0);
            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }

        vec4 wx = cubic_weights(f.x);
        vec4 wy = cubic_weights(f.y);
        vec4 result = vec4(0.0);

        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                vec4 value = texelFetch(
                    source_tex,
                    clamp_pixel(base + ivec2(x - 1, y - 1)),
                    0
                );
                result += value * wx[x] * wy[y];
            }
        }

        return result;
    }

    void main()
    {
        ivec2 p = ivec2(gl_GlobalInvocationID.xy);
        if (p.x >= width || p.y >= height)
            return;

        int index = p.y * width + p.x;
        int seed = nearest_map[index];

        if (seed < 0) {
            imageStore(output_img, p, vec4(0.0));
            return;
        }

        if (fixed_padding != 0) {
            ivec2 s = ivec2(seed % width, seed / width);
            ivec2 d = s - p;

            if (d.x * d.x + d.y * d.y > padding * padding) {
                imageStore(output_img, p, vec4(0.0));
                return;
            }
        }

        ivec2 seed_pos = ivec2(seed % width, seed / width);
        vec2 uv = texelFetch(coords_tex, seed_pos, 0).rg;

        imageStore(output_img, p, sample_source(uv));
    }
    """)

    _SAMPLE_SHADER = gpu.shader.create_from_info(info)
    return _SAMPLE_SHADER



def _get_shader(gpu):
    global _SHADER
    if _SHADER is not None:
        return _SHADER

    info = gpu.types.GPUShaderCreateInfo()
    info.storage_buf(0, "int", "src[]", qualifiers={'READ'})
    info.storage_buf(1, "int", "dst[]", qualifiers={'WRITE'})
    info.push_constant('INT', "width")
    info.push_constant('INT', "height")
    info.push_constant('INT', "step")
    info.local_group_size(16, 16)

    info.compute_source("""
    void main()
    {
        ivec2 p = ivec2(gl_GlobalInvocationID.xy);
        if (p.x >= width || p.y >= height) return;

        int index = p.y * width + p.x;
        int best = src[index];
        uint best_dist = 0xffffffffu;

        if (best >= 0) {
            ivec2 s = ivec2(best % width, best / width);
            ivec2 d = s - p;
            best_dist = uint(d.x * d.x + d.y * d.y);
        }

        for (int oy = -1; oy <= 1; oy++) {
            for (int ox = -1; ox <= 1; ox++) {
                if (ox == 0 && oy == 0) continue;

                ivec2 q = p + ivec2(ox, oy) * step;
                if (q.x < 0 || q.x >= width || q.y < 0 || q.y >= height)
                    continue;

                int candidate = src[q.y * width + q.x];
                if (candidate < 0) continue;

                ivec2 s = ivec2(candidate % width, candidate / width);
                ivec2 d = s - p;
                uint dist = uint(d.x * d.x + d.y * d.y);

                if (dist < best_dist) {
                    best = candidate;
                    best_dist = dist;
                }
            }
        }

        dst[index] = best;
    }
    """)

    _SHADER = gpu.shader.create_from_info(info)
    return _SHADER


def _dispatch(shader, src, dst, width, height, step):
    shader.bind()
    try:
        shader.storage_block("src", src)
        shader.storage_block("dst", dst)
        shader.uniform_int("width", width)
        shader.uniform_int("height", height)
        shader.uniform_int("step", step)

        gpu.compute.dispatch(
            shader,
            (width + 15) // 16,
            (height + 15) // 16,
            1,
        )
    finally:
        gpu.shader.unbind()


def jump_flood(seed_map, width, height, gpu_input=False):
    if gpu_input:
        src = seed_map
        resources = [seed_map]
    else:
        data = np.ascontiguousarray(seed_map, dtype=np.int32).reshape(-1)
        src = gpu.types.GPUStorageBuf(
            gpu.types.Buffer('INT', (data.size,), data)
        )
        resources = [src]

    count = width * height
    dst = gpu.types.GPUStorageBuf(
        gpu.types.Buffer('INT', (count,), np.empty(count, dtype=np.int32))
    )
    resources.append(dst)

    shader = _get_shader(gpu)
    step = 1 << max(0, (max(width, height) - 1).bit_length() - 1)

    while step:
        _dispatch(shader, src, dst, width, height, step)
        src, dst = dst, src
        step //= 2

    _dispatch(shader, src, dst, width, height, 1)
    src, dst = dst, src

    return src, resources


def create_sampling_maps(mapping):
    w, h = mapping.width, mapping.height

    if mapping.gpu_source_coords is None:
        if mapping.source_coords is None:
            raise RuntimeError("Source coordinates are unavailable")

        coords = np.ascontiguousarray(
            mapping.source_coords, dtype=np.float32
        ).reshape(-1)

        buffer = gpu.types.Buffer('FLOAT', (coords.size,), coords)
        mapping.gpu_source_coords = gpu.types.GPUTexture(
            (w, h), format='RG32F', data=buffer
        )

    if mapping.gpu_nearest_map is None:
        if mapping.nearest_map is None:
            raise RuntimeError("Nearest map is unavailable")

        nearest = np.ascontiguousarray(
            mapping.nearest_map, dtype=np.int32
        ).reshape(-1)

        buffer = gpu.types.Buffer('INT', (nearest.size,), nearest)
        mapping.gpu_nearest_map = gpu.types.GPUStorageBuf(buffer)


def reproject_image(image, mapping, settings):
    import bpy
    from .core import ReprojectionError

    w, h = mapping.width, mapping.height
    timings = {}

    try:
        start = perf_counter()

        if image.source == 'FILE' and not image.has_data:
            image.reload()

        source_pixels = np.empty((h, w, 4), dtype=np.float32)
        image.pixels.foreach_get(source_pixels.ravel())
        timings["read"] = perf_counter() - start

        start = perf_counter()

        source_buffer = gpu.types.Buffer(
            'FLOAT',
            (source_pixels.size,),
            source_pixels.ravel(),
        )
        source_texture = gpu.types.GPUTexture(
            (w, h),
            format='RGBA32F',
            data=source_buffer,
        )
        output_texture = gpu.types.GPUTexture(
            (w, h),
            format='RGBA32F',
        )
        timings["upload"] = perf_counter() - start

        interpolation = {
            'NEAREST': 0,
            'LINEAR': 1,
            'CUBIC': 2,
        }[settings.interpolation]

        shader = _get_sample_shader(gpu)
        shader.bind()

        try:
            shader.uniform_sampler("source_tex", source_texture)
            shader.uniform_sampler("coords_tex", mapping.gpu_source_coords)
            shader.storage_block("nearest_map", mapping.gpu_nearest_map)
            shader.image("output_img", output_texture)
            shader.uniform_int("width", w)
            shader.uniform_int("height", h)
            shader.uniform_int("interpolation", interpolation)
            shader.uniform_int(
                "fixed_padding", int(settings.dilation_mode == 'FIXED')
            )
            shader.uniform_int("padding", settings.padding)

            start = perf_counter()
            gpu.compute.dispatch(
                shader,
                (w + 15) // 16,
                (h + 15) // 16,
                1,
            )
            timings["sample"] = perf_counter() - start

        finally:
            gpu.shader.unbind()

        start = perf_counter()
        framebuffer = gpu.types.GPUFrameBuffer(
            color_slots=(output_texture,)
        )
        read_buffer = gpu.types.Buffer('FLOAT', (w * h * 4,))

        with framebuffer.bind():
            framebuffer.read_color(
                0, 0, w, h, 4, 0, 'FLOAT', data=read_buffer
            )

        pixels = np.frombuffer(
            bytes(read_buffer), dtype=np.float32
        ).copy()
        timings["readback"] = perf_counter() - start

        from .core import check_gpu_overlap
        check_gpu_overlap(mapping, settings)
        mapping.gpu_resources.clear()

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
            result.pixels.foreach_set(pixels)
            result.update()
            timings["write"] = perf_counter() - start

        except Exception:
            bpy.data.images.remove(result)
            raise

        timings["backend"] = "GPU"
        result["utr_source_image"] = image.name
        result["utr_output_suffix"] = settings.output_suffix
        return result, timings

    except Exception as exc:
        if isinstance(exc, ReprojectionError):
            raise
        raise ReprojectionError(
            f"Cannot process {image.name}: {exc}"
        ) from exc
