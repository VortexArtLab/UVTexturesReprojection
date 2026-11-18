
import numpy as np


_SHADER = None


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


def _dispatch(gpu, shader, src, dst, width, height, step):
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
        _dispatch(gpu, shader, src, dst, w, h, step)
        src, dst = dst, src
        step //= 2

    _dispatch(gpu, shader, src, dst, w, h, 1)
    src, dst = dst, src

    return np.frombuffer(bytes(src.read()), dtype=np.int32).reshape(h, w)
