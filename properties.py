
import bpy
from bpy.props import (
    StringProperty,
    IntProperty,
    EnumProperty,
    PointerProperty,
    CollectionProperty,
)


class UTR_PG_Texture(bpy.types.PropertyGroup):
    image: PointerProperty(name="Image", type=bpy.types.Image)


class UTR_PG_Settings(bpy.types.PropertyGroup):
    source_uv: StringProperty(name="Source UV")
    destination_uv: StringProperty(name="Destination UV")

    textures: CollectionProperty(type=UTR_PG_Texture)
    active_texture_index: IntProperty(default=0, min=0)

    compute_backend: EnumProperty(
        name="Compute Backend",
        items=[
            ('AUTO', "Auto", "Use GPU with CPU fallback"),
            ('CPU', "CPU", "Use CPU implementation"),
            ('GPU', "GPU", "Require GPU compute"),
        ],
        default='AUTO',
    )

    interpolation: EnumProperty(
        name="Interpolation",
        items=[
            ('NEAREST', "Nearest", ""),
            ('LINEAR', "Linear", ""),
            ('CUBIC', "Cubic", ""),
        ],
        default='LINEAR',
    )

    dilation_mode: EnumProperty(
        name="Dilation",
        items=[
            ('BOUNDS', "Image Bounds", "Fill the entire image using nearest UV island pixels"),
            ('FIXED', "Fixed Padding", "Extend UV islands by a fixed pixel radius"),
        ],
        default='BOUNDS',
    )

    padding: IntProperty(name="Padding", default=8, min=0, max=256)

    overlap_policy: EnumProperty(
        name="Overlaps",
        items=[
            ('ABORT', "Abort", ""),
            ('OVERWRITE', "Overwrite", ""),
        ],
        default='ABORT',
    )

    output_suffix: StringProperty(name="Output Suffix", default="_reprojected")


CLASSES = (
    UTR_PG_Texture,
    UTR_PG_Settings,
)
