
import bpy
from .models import ReprojectionError


def collect_material_images(obj):
    images = []
    visited = set()
    found = set()

    def visit(tree):
        if not tree:
            return
        pointer = tree.as_pointer()
        if pointer in visited:
            return
        visited.add(pointer)

        for node in tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                image = node.image
                pointer = image.as_pointer()
                if pointer not in found:
                    found.add(pointer)
                    images.append(image)
            elif node.type == 'GROUP':
                visit(node.node_tree)

    for slot in obj.material_slots:
        material = slot.material
        if material and material.use_nodes:
            visit(material.node_tree)

    return images


def iter_material_node_trees(obj):
    visited = set()
    stack = [
        slot.material.node_tree
        for slot in obj.material_slots
        if slot.material and slot.material.use_nodes and slot.material.node_tree
    ]

    while stack:
        tree = stack.pop()
        pointer = tree.as_pointer()
        if pointer in visited:
            continue

        visited.add(pointer)
        yield tree

        for node in tree.nodes:
            if node.type == 'GROUP' and node.node_tree:
                stack.append(node.node_tree)


def _match_socket(sockets, reference):
    identifier = getattr(reference, "identifier", "")
    if identifier:
        for socket in sockets:
            if socket.identifier == identifier:
                return socket

    return next(
        (socket for socket in sockets if socket.name == reference.name),
        None,
    )


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

            output = _match_socket(
                output_node.inputs, link.from_socket
            )

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
            input_socket = _match_socket(
                group_instance.inputs, link.from_socket
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


def collect_normal_map_images(obj, selected_images=None):
    selected = None
    if selected_images is not None:
        selected = {
            image.as_pointer()
            for image in selected_images
            if image is not None
        }

    found = {}

    def visit(tree, group_stack, ancestry):
        if not tree:
            return

        pointer = tree.as_pointer()
        if pointer in ancestry:
            return

        ancestry = ancestry | {pointer}

        for node in tree.nodes:
            if node.bl_idname == 'ShaderNodeNormalMap':
                color = node.inputs.get("Color")
                if color and color.is_linked:
                    images = _find_upstream_images(
                        color, group_stack, set()
                    )

                    active_uv = (
                        obj.data.uv_layers.active_render
                        or obj.data.uv_layers.active
                    )
                    uv_map = node.uv_map or (
                        active_uv.name if active_uv else ""
                    )

                    usage = (
                        node.space,
                        node.convention if node.space == 'TANGENT' else None,
                        uv_map if node.space == 'TANGENT' else None,
                    )

                    for image_node in images.values():
                        image = image_node.image
                        image_pointer = image.as_pointer()

                        if selected is not None and image_pointer not in selected:
                            continue

                        entry = found.setdefault(image_pointer, {
                            "image": image,
                            "uses": [],
                            "nodes": [],
                            "node_pointers": set(),
                        })

                        entry["uses"].append(usage)

                        node_pointer = node.as_pointer()
                        if node_pointer not in entry["node_pointers"]:
                            entry["node_pointers"].add(node_pointer)
                            entry["nodes"].append(node)

            if node.type == 'GROUP' and node.node_tree:
                visit(
                    node.node_tree,
                    group_stack + (node,),
                    ancestry,
                )

    for slot in obj.material_slots:
        material = slot.material
        if material and material.use_nodes:
            visit(material.node_tree, (), frozenset())

    for entry in found.values():
        uses = set(entry["uses"])

        if len(uses) != 1:
            raise ReprojectionError(
                f"Image '{entry['image'].name}' is used by "
                "incompatible Normal Map nodes"
            )

        space, convention, uv_map = next(iter(uses))

        entry["space"] = space
        entry["convention"] = convention
        entry["uv_map"] = uv_map

        del entry["uses"]
        del entry["node_pointers"]

    return found


def find_reprojected_image(image, suffix):
    matches = [
        candidate for candidate in bpy.data.images
        if candidate.get("utr_source_image") == image.name
        and candidate.get("utr_output_suffix") == suffix
    ]

    if matches:
        return matches[-1]

    return bpy.data.images.get(f"{image.name}{suffix}")


def update_materials(obj, settings):
    uv_names = {uv.name for uv in obj.data.uv_layers}
    if settings.destination_uv not in uv_names:
        raise ReprojectionError("Invalid destination UV map")

    normal_maps = collect_normal_map_images(obj)
    replacements = {}
    missing = []

    for item in settings.textures:
        source = item.image
        if source is None:
            continue

        output = find_reprojected_image(source, settings.output_suffix)

        if output is None:
            missing.append(source.name)
            continue

        replacements[source.as_pointer()] = output
        replacements[output.as_pointer()] = output

    if not replacements:
        return 0, missing

    updated_nodes = 0

    for tree in iter_material_node_trees(obj):
        targets = []

        for node in tree.nodes:
            if node.type != 'TEX_IMAGE' or node.image is None:
                continue

            output = replacements.get(node.image.as_pointer())
            if output is None:
                continue

            node.image = output
            targets.append(node)

        if not targets:
            continue

        target_sockets = {
            node.inputs["Vector"].as_pointer()
            for node in targets
            if node.inputs.get("Vector") is not None
        }

        existing_uv = {}

        for node in targets:
            vector = node.inputs.get("Vector")
            if vector is None:
                continue

            for link in vector.links:
                source = link.from_node
                if source.bl_idname == 'ShaderNodeUVMap':
                    existing_uv[source.as_pointer()] = source

        reusable_uv = []

        for uv_node in existing_uv.values():
            output = uv_node.outputs.get("UV")
            if output is None:
                continue

            if all(
                link.to_socket.as_pointer() in target_sockets
                for link in output.links
            ):
                reusable_uv.append(uv_node)

        if reusable_uv:
            for uv_node in reusable_uv:
                uv_node.uv_map = settings.destination_uv
            shared_uv = reusable_uv[0]
        else:
            shared_uv = tree.nodes.new("ShaderNodeUVMap")
            shared_uv.uv_map = settings.destination_uv
            shared_uv.label = "Destination UV"

            texture_x = min(node.location.x for node in targets)
            texture_y = sum(node.location.y for node in targets) / len(targets)
            shared_uv.location = (texture_x - 220, texture_y)
            reusable_uv = [shared_uv]

        for node in targets:
            vector = node.inputs.get("Vector")
            if vector is None:
                continue

            if any(link.from_node in reusable_uv for link in vector.links):
                continue

            for link in list(vector.links):
                tree.links.remove(link)

            tree.links.new(shared_uv.outputs["UV"], vector)

        updated_nodes += len(targets)

    updated_normal_nodes = set()

    for pointer, entry in normal_maps.items():
        if pointer not in replacements:
            continue
        if entry["space"] != 'TANGENT':
            continue

        for node in entry["nodes"]:
            node_pointer = node.as_pointer()
            if node_pointer in updated_normal_nodes:
                continue

            node.uv_map = settings.destination_uv
            updated_normal_nodes.add(node_pointer)

    return updated_nodes, missing
