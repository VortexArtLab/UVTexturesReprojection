
import numpy as np
from time import perf_counter

from .constants import TILE_SIZE
from .models import ReprojectionError, UVOverlapError


def _top_left(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return dy > 0.0 or (abs(dy) < 1e-12 and dx < 0.0)


def _rasterize_triangle(
    source_coords, seed_map, src, dst, settings,
    pixel_grid, tangent_map=None, tangent_triangle=None,
):
    h, w = seed_map.shape
    x_indices, y_indices, x_centers, y_centers, scale = pixel_grid
    dst = dst * scale

    area = (
        (dst[1, 0] - dst[0, 0]) * (dst[2, 1] - dst[0, 1]) -
        (dst[1, 1] - dst[0, 1]) * (dst[2, 0] - dst[0, 0])
    )

    if abs(area) < 1e-10:
        return

    if area < 0.0:
        dst = dst[[0, 2, 1]]
        src = src[[0, 2, 1]]

        if tangent_triangle is not None:
            tangent_triangle = tangent_triangle[[0, 2, 1]]

        area = -area

    x0 = max(0, int(np.ceil(dst[:, 0].min() - 0.5)))
    x1 = min(w - 1, int(np.floor(dst[:, 0].max() - 0.5)))
    y0 = max(0, int(np.ceil(dst[:, 1].min() - 0.5)))
    y1 = min(h - 1, int(np.floor(dst[:, 1].max() - 0.5)))

    if x0 > x1 or y0 > y1:
        return

    a, b, c = dst
    top_left_0 = _top_left(a, b)
    top_left_1 = _top_left(b, c)
    top_left_2 = _top_left(c, a)
    eps = 1e-7

    for ty in range(y0, y1 + 1, TILE_SIZE):
        ty1 = min(ty + TILE_SIZE, y1 + 1)

        for tx in range(x0, x1 + 1, TILE_SIZE):
            tx1 = min(tx + TILE_SIZE, x1 + 1)
            tile_shape = (ty1 - ty, tx1 - tx)

            xx = np.broadcast_to(
                x_indices[None, tx:tx1],
                tile_shape,
            )
            yy = np.broadcast_to(
                y_indices[ty:ty1, None],
                tile_shape,
            )

            px = x_centers[None, tx:tx1]
            py = y_centers[ty:ty1, None]

            e0 = (
                (b[0] - a[0]) * (py - a[1])
                - (b[1] - a[1]) * (px - a[0])
            )
            e1 = (
                (c[0] - b[0]) * (py - b[1])
                - (c[1] - b[1]) * (px - b[0])
            )
            e2 = (
                (a[0] - c[0]) * (py - c[1])
                - (a[1] - c[1]) * (px - c[0])
            )

            inside = (
                ((e0 > eps) | ((np.abs(e0) <= eps) & top_left_0)) &
                ((e1 > eps) | ((np.abs(e1) <= eps) & top_left_1)) &
                ((e2 > eps) | ((np.abs(e2) <= eps) & top_left_2))
            )

            if not np.any(inside):
                continue

            seeds = seed_map[ty:ty1, tx:tx1]

            if (
                settings.overlap_policy == 'ABORT'
                and np.any(inside & (seeds >= 0))
            ):
                raise UVOverlapError("Destination UV overlap detected")

            w0 = e1[inside] / area
            w1 = e2[inside] / area
            w2 = e0[inside] / area

            uv = (
                w0[:, None] * src[0]
                + w1[:, None] * src[1]
                + w2[:, None] * src[2]
            )

            source_coords[ty:ty1, tx:tx1][inside] = uv
            seeds[inside] = yy[inside] * w + xx[inside]

            if tangent_map is not None:
                matrices = (
                    w0[:, None] * tangent_triangle[0]
                    + w1[:, None] * tangent_triangle[1]
                    + w2[:, None] * tangent_triangle[2]
                )

                axis_x = np.stack(
                    (matrices[:, 0], matrices[:, 2]), axis=1
                )
                axis_length = np.linalg.norm(axis_x, axis=1)

                determinant = (
                    matrices[:, 0] * matrices[:, 3]
                    - matrices[:, 1] * matrices[:, 2]
                )

                handedness = np.where(determinant < 0.0, -1.0, 1.0)
                axis_x /= np.maximum(axis_length[:, None], 1e-8)

                orthogonal = np.empty_like(matrices)
                orthogonal[:, 0] = axis_x[:, 0]
                orthogonal[:, 1] = -axis_x[:, 1] * handedness
                orthogonal[:, 2] = axis_x[:, 1]
                orthogonal[:, 3] = axis_x[:, 0] * handedness

                tangent_map[ty:ty1, tx:tx1][inside] = orthogonal


def rasterize_cpu(triangles, width, height, settings):
    source_triangles = triangles.source_triangles
    destination_triangles = triangles.destination_triangles
    tangent_triangles = triangles.tangent_transforms

    source_coords = np.zeros(
        (height, width, 2), dtype=np.float32
    )
    seed_map = np.full(
        (height, width), -1, dtype=np.int32
    )

    tangent_map = None

    if tangent_triangles is not None:
        tangent_map = np.zeros(
            (height, width, 4), dtype=np.float32
        )

    pixel_grid = (
        np.arange(width, dtype=np.int32),
        np.arange(height, dtype=np.int32),
        np.arange(width, dtype=np.float32) + 0.5,
        np.arange(height, dtype=np.float32) + 0.5,
        np.array((width, height), dtype=np.float32),
    )

    start = perf_counter()

    for index in range(len(source_triangles)):
        _rasterize_triangle(
            source_coords,
            seed_map,
            source_triangles[index],
            destination_triangles[index],
            settings,
            pixel_grid,
            tangent_map,
            tangent_triangles[index] if tangent_triangles is not None else None,
        )

    raster_time = perf_counter() - start

    if not np.any(seed_map >= 0):
        raise ReprojectionError("No pixels generated")

    return source_coords, seed_map, tangent_map, {"cpu": raster_time}
