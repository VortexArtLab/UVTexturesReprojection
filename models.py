
from typing import NamedTuple

import numpy as np


class ReprojectionError(Exception):
    pass


class UVOverlapError(ReprojectionError):
    pass


class TriangleData(NamedTuple):
    source_triangles: np.ndarray
    destination_triangles: np.ndarray
    tangent_transforms: np.ndarray | None
    normal_maps: dict


class ReprojectionMap:
    __slots__ = (
        "width", "height",
        "source_coords", "nearest_map",
        "tangent_map", "normal_maps",
        "gpu_source_coords", "gpu_nearest_map",
        "gpu_tangent_map", "gpu_overlap_flag",
        "gpu_resources", "gpu_error", "overlap_checked",
    )

    def __init__(
        self, width, height, source_coords, nearest_map,
        tangent_map=None, normal_maps=None,
    ):
        self.width = width
        self.height = height
        self.source_coords = source_coords
        self.nearest_map = nearest_map
        self.tangent_map = tangent_map
        self.normal_maps = normal_maps or {}
        self.gpu_source_coords = None
        self.gpu_nearest_map = None
        self.gpu_tangent_map = None
        self.gpu_overlap_flag = None
        self.gpu_resources = []
        self.gpu_error = ""
        self.overlap_checked = False
