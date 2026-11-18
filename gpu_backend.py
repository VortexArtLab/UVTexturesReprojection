
import numpy as np
import gpu


_SHADER = None
_SAMPLE_SHADER = None


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


def jump_flood(seed_map):
    import gpu

    h, w = seed_map.shape
    count = w * h

    src_data = gpu.types.Buffer(
        'INT', (count,), np.ascontiguousarray(seed_map).reshape(-1)
    )
    dst_data = gpu.types.Buffer(
        'INT', (count,), np.empty(count, dtype=np.int32)
    )

    src = gpu.types.GPUStorageBuf(src_data)
    dst = gpu.types.GPUStorageBuf(dst_data)
    shader = _get_shader(gpu)

    step = 1 << max(0, (max(w, h) - 1).bit_length() - 1)

    while step:
        _dispatch(shader, src, dst, w, h, step)
        src, dst = dst, src
        step //= 2

    _dispatch(shader, src, dst, w, h, 1)
    src, dst = dst, src

    return src


def create_sampling_maps(mapping):
    import gpu

    w, h = mapping.width, mapping.height

    coords = np.ascontiguousarray(
        mapping.source_coords, dtype=np.float32
    ).reshape(-1)

    coords_buffer = gpu.types.Buffer(
        'FLOAT', (coords.size,), coords
    )
    source_coords = gpu.types.GPUTexture(
        (w, h), format='RG32F', data=coords_buffer
    )

    nearest_map = mapping.gpu_nearest_map

    if nearest_map is None:
        if mapping.nearest_map is None:
            raise RuntimeError("Nearest map is unavailable")

        nearest = np.ascontiguousarray(
            mapping.nearest_map, dtype=np.int32
        ).reshape(-1)

        nearest_buffer = gpu.types.Buffer(
            'INT', (nearest.size,), nearest
        )
        nearest_map = gpu.types.GPUStorageBuf(nearest_buffer)

    mapping.gpu_source_coords = source_coords
    mapping.gpu_nearest_map = nearest_map


def reproject_image(image, mapping, settings):
    import bpy
    import gpu
    from time import perf_counter
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
        return result, timings

    except Exception as exc:
        if isinstance(exc, ReprojectionError):
            raise
        raise ReprojectionError(
            f"Cannot process {image.name}: {exc}"
        ) from exc
