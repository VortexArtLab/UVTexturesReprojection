import bpy
import numpy as np
from time import perf_counter
from . import geometry
from .models import (
    ReprojectionError,
    UVOverlapError,
    ReprojectionMap,
    TriangleData,
)
from . import materials
from .materials import (
    collect_material_images,
    collect_normal_map_images,
    find_reprojected_image,
    iter_material_node_trees,
    update_materials,
)
from .constants import TILE_SIZE
from . import raster
from . import jfa
from . import mapping, sampling
from .mapping import (
    build_reprojection_map,
    build_uv_triangles,
    ensure_cpu_mapping,
)
from .validation import validate_settings


def _find_upstream_images(socket, group_stack, visited):
    images = {}
    context = tuple(node.as_pointer() for node in group_stack)
    key = (socket.as_pointer(), context)

    if key in visited:
        return images

    visited.add(key)

    def add(found):
        images.update(found)

    for link in socket.links:
        node = link.from_node

        if node.type == 'TEX_IMAGE':
            if node.image:
                images[node.image.as_pointer()] = node

        elif node.type == 'REROUTE':
            add(_find_upstream_images(
                node.inputs[0],
                group_stack,
                visited,
            ))

        elif node.type == 'GROUP':
            tree = node.node_tree
            if not tree:
                continue

            outputs = [
                item for item in tree.nodes
                if item.type == 'GROUP_OUTPUT'
            ]
            output_node = next(
                (item for item in outputs if item.is_active_output),
                outputs[0] if outputs else None,
            )

            if output_node is None:
                continue

            output = output_node.inputs.get(link.from_socket.identifier)
            if output is None:
                output = output_node.inputs.get(link.from_socket.name)

            if output is not None:
                add(_find_upstream_images(
                    output,
                    group_stack + (node,),
                    visited,
                ))

        elif node.type == 'GROUP_INPUT':
            if not group_stack:
                continue

            group_instance = group_stack[-1]
            input_socket = group_instance.inputs.get(
                link.from_socket.identifier
            )
            if input_socket is None:
                input_socket = group_instance.inputs.get(
                    link.from_socket.name
                )

            if input_socket is not None:
                add(_find_upstream_images(
                    input_socket,
                    group_stack[:-1],
                    visited,
                ))

        else:
            for input_socket in node.inputs:
                if (
                    input_socket.is_linked
                    and input_socket.type in {'RGBA', 'VECTOR'}
                ):
                    add(_find_upstream_images(
                        input_socket,
                        group_stack,
                        visited,
                    ))

    return images


def reproject_image(image, mapping, settings):
    map_readback = 0.0

    if settings.compute_backend != 'CPU':
        gpu_ready = (
            mapping.gpu_source_coords is not None
            and mapping.gpu_nearest_map is not None
        )

        if gpu_ready:
            try:
                from . import gpu_backend
                result, timings = gpu_backend.reproject_image(
                    image, mapping, settings
                )
                timings["backend"] = "GPU"
                return result, timings

            except UVOverlapError:
                raise

            except Exception as exc:
                if settings.compute_backend == 'GPU':
                    raise ReprojectionError(
                        f"GPU sampling failed: {exc}"
                    ) from exc

                mapping.gpu_error = "; ".join(filter(
                    None, (mapping.gpu_error, f"Sampling: {exc}")
                ))

        elif settings.compute_backend == 'GPU':
            raise ReprojectionError(
                f"GPU sampling unavailable: "
                f"{mapping.gpu_error or 'GPU maps missing'}"
            )

        map_readback = ensure_cpu_mapping(mapping, settings)

    result, timings = sampling._reproject_image_cpu(image, mapping, settings)
    timings["map_readback"] = map_readback

    if settings.compute_backend == 'AUTO' and mapping.gpu_error:
        timings["backend"] = "CPU fallback"
        timings["fallback_reason"] = mapping.gpu_error
    else:
        timings["backend"] = "CPU"

    return result, timings
