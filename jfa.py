
import numpy as np
from .constants import TILE_SIZE


def _jump_flood_pass(
    current,
    target,
    width,
    height,
    step,
    x_coords,
    y_coords,
    workspace,
):
    work_x, work_y, best_dist, candidate_dist, valid, update = workspace

    for y0 in range(0, height, TILE_SIZE):
        y1 = min(y0 + TILE_SIZE, height)
        tile_height = y1 - y0
        ys = y_coords[y0:y1, None]

        for x0 in range(0, width, TILE_SIZE):
            x1 = min(x0 + TILE_SIZE, width)
            tile_width = x1 - x0
            xs = x_coords[None, x0:x1]

            best = target[y0:y1, x0:x1]
            np.copyto(best, current[y0:y1, x0:x1])

            wx = work_x[:tile_height, :tile_width]
            wy = work_y[:tile_height, :tile_width]
            bd = best_dist[:tile_height, :tile_width]
            mask = valid[:tile_height, :tile_width]

            np.remainder(best, width, out=wx)
            np.floor_divide(best, width, out=wy)
            np.subtract(wx, xs, out=wx)
            np.square(wx, out=wx)
            np.subtract(wy, ys, out=wy)
            np.square(wy, out=wy)
            np.add(wx, wy, out=bd)

            np.less(best, 0, out=mask)
            np.copyto(bd, np.iinfo(bd.dtype).max, where=mask)

            for oy in (-step, 0, step):
                ay = max(y0, -oy)
                by = min(y1, height - oy)

                if ay >= by:
                    continue

                for ox in (-step, 0, step):
                    if ox == 0 and oy == 0:
                        continue

                    ax = max(x0, -ox)
                    bx = min(x1, width - ox)

                    if ax >= bx:
                        continue

                    candidates = current[
                        ay + oy:by + oy,
                        ax + ox:bx + ox,
                    ]

                    candidate_height, candidate_width = candidates.shape

                    wx = work_x[:candidate_height, :candidate_width]
                    wy = work_y[:candidate_height, :candidate_width]
                    dist = candidate_dist[:candidate_height, :candidate_width]
                    mask = valid[:candidate_height, :candidate_width]
                    update_mask = update[:candidate_height, :candidate_width]

                    np.greater_equal(candidates, 0, out=mask)

                    if not mask.any():
                        continue

                    np.remainder(candidates, width, out=wx)
                    np.floor_divide(candidates, width, out=wy)

                    np.subtract(
                        wx,
                        x_coords[None, ax:bx],
                        out=wx,
                    )
                    np.square(wx, out=wx)

                    np.subtract(
                        wy,
                        y_coords[ay:by, None],
                        out=wy,
                    )
                    np.square(wy, out=wy)

                    np.add(wx, wy, out=dist)

                    sy = slice(ay - y0, by - y0)
                    sx = slice(ax - x0, bx - x0)

                    best_sub = best[sy, sx]
                    dist_sub = bd[sy, sx]

                    np.less(dist, dist_sub, out=update_mask)
                    np.logical_and(
                        update_mask,
                        mask,
                        out=update_mask,
                    )

                    np.copyto(best_sub, candidates, where=update_mask)
                    np.copyto(dist_sub, dist, where=update_mask)


def jump_flood(seed_map):
    height, width = seed_map.shape

    if seed_map.dtype.kind != "i":
        raise TypeError("Seed map must use a signed integer dtype")

    max_distance = (width - 1) ** 2 + height ** 2

    use_int32 = (
        seed_map.dtype.itemsize <= 4
        and max_distance <= np.iinfo(np.int32).max
    )
    dtype = np.int32 if use_int32 else np.int64

    current = seed_map
    buffer = np.empty_like(current)

    x_coords = np.arange(width, dtype=dtype)
    y_coords = np.arange(height, dtype=dtype)

    workspace_shape = (
        min(TILE_SIZE, height),
        min(TILE_SIZE, width),
    )

    workspace = tuple(
        np.empty(workspace_shape, dtype=dtype)
        for _ in range(4)
    ) + tuple(
        np.empty(workspace_shape, dtype=np.bool_)
        for _ in range(2)
    )

    step = 1 << max(
        0,
        (max(width, height) - 1).bit_length() - 1,
    )

    while step:
        _jump_flood_pass(
            current,
            buffer,
            width,
            height,
            step,
            x_coords,
            y_coords,
            workspace,
        )
        current, buffer = buffer, current
        step //= 2

    _jump_flood_pass(
        current,
        buffer,
        width,
        height,
        1,
        x_coords,
        y_coords,
        workspace,
    )
    current, buffer = buffer, current

    return current
