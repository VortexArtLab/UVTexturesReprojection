def validate_settings(obj):
    if obj is None or obj.type != 'MESH':
        return False, "Active object must be a mesh"

    if obj.mode != 'OBJECT':
        return False, "Object must be in Object Mode"

    settings = obj.utr_settings
    uv_names = {uv.name for uv in obj.data.uv_layers}

    if settings.source_uv not in uv_names:
        return False, "Invalid source UV map"

    if settings.destination_uv not in uv_names:
        return False, "Invalid destination UV map"

    if settings.source_uv == settings.destination_uv:
        return False, "Source and destination must be different"

    if not obj.data.polygons:
        return False, "Mesh has no polygons"

    if not settings.textures:
        return False, "Add at least one texture"

    seen = set()
    for item in settings.textures:
        image = item.image
        if image is None:
            return False, "Texture slot has no image"
        if image.as_pointer() in seen:
            return False, f"Duplicate image: {image.name}"
        seen.add(image.as_pointer())
        if image.source not in {'FILE', 'GENERATED'}:
            return False, f"Unsupported image source: {image.name}"
        if image.is_multiview:
            return False, f"Multiview image is unsupported: {image.name}"
        if image.size[0] < 1 or image.size[1] < 1:
            return False, f"Invalid image dimensions: {image.name}"

    return True, ""