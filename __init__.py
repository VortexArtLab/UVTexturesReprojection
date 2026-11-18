
import bpy
from bpy.props import PointerProperty
from . import properties, operators, panel

CLASSES = (
    *properties.CLASSES,
    *operators.CLASSES,
    *panel.CLASSES,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.utr_settings = PointerProperty(type=properties.UTR_PG_Settings)


def unregister():
    del bpy.types.Object.utr_settings
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
