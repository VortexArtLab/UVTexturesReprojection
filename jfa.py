
import numpy as np

from .constants import TILE_SIZE


def _jump_flood_pass(current, target, width, height, step):
    for y0 in range(0, height, TILE_SIZE):
        y1 = min(y0 + TILE_SIZE, height)

        for x0 in range(0, width, TILE_SIZE):
            x1 = min(x0 + TILE_SIZE, width)

            best = current[y0:y1, x0:x1].copy()
            ys = np.arange(y0, y1, dtype=np.int64)[:, None]
            xs = np.arange(x0, x1, dtype=np.int64)[None, :]
            valid = best >= 0
            seed_x = np.where(valid, best % width, 0).astype(np.int64)
            seed_y = np.where(valid, best // width, 0).astype(np.int64)

            dx = seed_x - xs
            dy = seed_y - ys
            best_dist = dx * dx + dy * dy
            best_dist[~valid] = np.iinfo(np.int64).max

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
                    valid = candidates >= 0

                    if not np.any(valid):
                        continue

                    cx = np.where(valid, candidates % width, 0).astype(np.int64)
                    cy = np.where(valid, candidates // width, 0).astype(np.int64)

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


def jump_flood(seed_map):
    height, width = seed_map.shape
    current = seed_map
    buffer = np.empty_like(current)

    step = 1 << max(
        0,
        (max(width, height) - 1).bit_length() - 1,
    )

    while step:
        _jump_flood_pass(current, buffer, width, height, step)
        current, buffer = buffer, current
        step //= 2

    _jump_flood_pass(current, buffer, width, height, 1)
    current, buffer = buffer, current

    return current
