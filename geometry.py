
import numpy as np

from .models import ReprojectionError, TriangleData


def _calc_loop_tangents(mesh, uv_name):
    calculated = False
    try:
        mesh.calc_tangents(uvmap=uv_name)
        calculated = True

        loop_count = len(mesh.loops)
        tangents = np.empty(loop_count * 3, dtype=np.float32)
        normals = np.empty(loop_count * 3, dtype=np.float32)
        signs = np.empty(loop_count, dtype=np.float32)

        mesh.loops.foreach_get("tangent", tangents)
        mesh.loops.foreach_get("normal", normals)
        mesh.loops.foreach_get("bitangent_sign", signs)

        return (
            tangents.reshape(-1, 3),
            normals.reshape(-1, 3),
            signs,
        )
    except Exception as exc:
        raise ReprojectionError(
            f"Cannot calculate tangents for UV map '{uv_name}': {exc}"
        ) from exc
    finally:
        if calculated:
            mesh.free_tangents()


def _build_tangent_transforms(mesh, loop_indices, source_uv, destination_uv):
    source_tangent, normals, source_sign = _calc_loop_tangents(
        mesh, source_uv
    )
    destination_tangent, _, destination_sign = _calc_loop_tangents(
        mesh, destination_uv
    )

    normal_length = np.linalg.norm(normals, axis=1)
    source_length = np.linalg.norm(source_tangent, axis=1)
    destination_length = np.linalg.norm(destination_tangent, axis=1)

    if (
        np.any(normal_length < 1e-6)
        or np.any(source_length < 1e-6)
        or np.any(destination_length < 1e-6)
    ):
        raise ReprojectionError(
            "Cannot build tangent transforms: degenerate tangent basis"
        )

    normals /= normal_length[:, None]
    source_tangent -= (
        normals * np.sum(source_tangent * normals, axis=1)[:, None]
    )
    destination_tangent -= (
        normals * np.sum(destination_tangent * normals, axis=1)[:, None]
    )

    source_tangent /= np.maximum(
        np.linalg.norm(source_tangent, axis=1)[:, None], 1e-8
    )
    destination_tangent /= np.maximum(
        np.linalg.norm(destination_tangent, axis=1)[:, None], 1e-8
    )

    source_sign = np.where(source_sign < 0.0, -1.0, 1.0)
    destination_sign = np.where(destination_sign < 0.0, -1.0, 1.0)

    source_bitangent = (
        np.cross(normals, source_tangent) * source_sign[:, None]
    )
    destination_bitangent = (
        np.cross(normals, destination_tangent)
        * destination_sign[:, None]
    )

    transforms = np.empty((len(mesh.loops), 4), dtype=np.float32)
    transforms[:, 0] = np.sum(
        destination_tangent * source_tangent, axis=1
    )
    transforms[:, 1] = np.sum(
        destination_tangent * source_bitangent, axis=1
    )
    transforms[:, 2] = np.sum(
        destination_bitangent * source_tangent, axis=1
    )
    transforms[:, 3] = np.sum(
        destination_bitangent * source_bitangent, axis=1
    )

    determinant = (
        transforms[:, 0] * transforms[:, 3]
        - transforms[:, 1] * transforms[:, 2]
    )
    if np.any(np.abs(determinant) < 0.5):
        raise ReprojectionError(
            "Invalid source/destination tangent transformation"
        )

    return np.ascontiguousarray(transforms[loop_indices])


def build_uv_triangles(obj, normal_maps):
    mesh = obj.data
    settings = obj.utr_settings

    mesh.calc_loop_triangles()
    triangle_count = len(mesh.loop_triangles)

    if not triangle_count:
        raise ReprojectionError("No triangles found")

    source_layer = mesh.uv_layers.get(settings.source_uv)
    destination_layer = mesh.uv_layers.get(settings.destination_uv)

    if source_layer is None or destination_layer is None:
        raise ReprojectionError("Source or destination UV map not found")

    loop_indices = np.empty(triangle_count * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", loop_indices)
    loop_indices = loop_indices.reshape(-1, 3)

    loop_count = len(mesh.loops)
    source_uv = np.empty(loop_count * 2, dtype=np.float32)
    destination_uv = np.empty(loop_count * 2, dtype=np.float32)

    source_layer.data.foreach_get("uv", source_uv)
    destination_layer.data.foreach_get("uv", destination_uv)

    source_uv = source_uv.reshape(-1, 2)
    destination_uv = destination_uv.reshape(-1, 2)

    source_triangles = np.ascontiguousarray(source_uv[loop_indices])
    destination_triangles = np.ascontiguousarray(
        destination_uv[loop_indices]
    )

    for name, uv in (
        ("Source", source_triangles),
        ("Destination", destination_triangles),
    ):
        if not np.all(np.isfinite(uv)):
            raise ReprojectionError(
                f"{name} UV coordinates contain invalid values"
            )
        if np.any(uv < -1e-6) or np.any(uv > 1.000001):
            raise ReprojectionError(
                f"{name} UV coordinates outside 0-1 are unsupported"
            )
        np.clip(uv, 0.0, 1.0, out=uv)

    tangent_maps = [
        entry for entry in normal_maps.values()
        if entry["space"] == 'TANGENT'
    ]

    for entry in tangent_maps:
        if not entry["image"].colorspace_settings.is_data:
            raise ReprojectionError(
                f"Normal map must use Non-Color: {entry['image'].name}"
            )

    tangent_transforms = None

    if tangent_maps:
        invalid_uvs = {
            entry["uv_map"]
            for entry in tangent_maps
            if entry["uv_map"] != settings.source_uv
        }

        if invalid_uvs:
            raise ReprojectionError(
                "Normal Map nodes use UV maps different from Source UV: "
                + ", ".join(sorted(invalid_uvs))
            )

        tangent_transforms = _build_tangent_transforms(
            mesh,
            loop_indices,
            settings.source_uv,
            settings.destination_uv,
        )

    return TriangleData(
        source_triangles,
        destination_triangles,
        tangent_transforms,
        normal_maps,
    )
