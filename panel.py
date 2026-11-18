
import bpy
from . import core

class UTR_UL_textures(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index, flt_flag):
        layout.prop(item, "image", text="", icon='IMAGE_DATA')


class UTR_MT_texture_options(bpy.types.Menu):
    bl_idname = "UTR_MT_texture_options"
    bl_label = "Texture Options"

    def draw(self, context):
        self.layout.operator(
            "utr.texture_get_from_materials",
            text="Get from Materials",
            icon='IMAGE_DATA',
        )

class UTR_PT_material(bpy.types.Panel):
    bl_label = "UV Textures Reprojection"
    bl_idname = "MATERIAL_PT_uv_textures_reprojection"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        return bool(context.object and context.object.type == 'MESH')

    def draw(self, context):
        layout = self.layout
        obj = context.object
        settings = obj.utr_settings

        layout.use_property_split = True
        layout.use_property_decorate = False
        uv_layout = layout.column(align=True)
        uv_layout.prop_search(settings, "source_uv", obj.data, "uv_layers")
        uv_layout.prop_search(settings, "destination_uv", obj.data, "uv_layers")

        layout.separator()
        layout.label(text="Textures")

        row = layout.row()
        row.template_list(
            "UTR_UL_textures", "",
            settings, "textures",
            settings, "active_texture_index",
            rows=3,
        )

        actions = row.column(align=True)
        actions.operator("utr.texture_add", text="", icon='ADD')

        remove = actions.row()
        remove.enabled = bool(settings.textures)
        remove.operator("utr.texture_remove", text="", icon='REMOVE')

        actions.separator()
        actions.menu("UTR_MT_texture_options", text="", icon='DOWNARROW_HLT')

        actions.separator()

        move_up = actions.row()
        move_up.enabled = settings.active_texture_index > 0
        move = move_up.operator("utr.texture_move", text="", icon='TRIA_UP')
        move.direction = 'UP'

        move_down = actions.row()
        move_down.enabled = settings.active_texture_index < len(settings.textures) - 1
        move = move_down.operator("utr.texture_move", text="", icon='TRIA_DOWN')
        move.direction = 'DOWN'

        layout.separator()
        layout.label(text="Processing")
        layout.prop(settings, "interpolation")
        layout.prop(settings, "dilation_mode")

        if settings.dilation_mode == 'FIXED':
            layout.prop(settings, "padding")

        layout.prop(settings, "overlap_policy")
        layout.prop(settings, "output_suffix")

        valid, message = core.validate_settings(obj)
        if not valid:
            layout.label(text=message, icon='INFO')

        layout.separator()
        row = layout.row()
        row.enabled = valid
        row.operator("utr.reproject", text="Reproject", icon='MOD_UVPROJECT')


CLASSES = (
    UTR_MT_texture_options,
    UTR_UL_textures,
    UTR_PT_material,
)