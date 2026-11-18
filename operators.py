
import bpy
from . import core
from bpy.props import EnumProperty, IntProperty


class UTR_OT_texture_move(bpy.types.Operator):
    bl_idname = "utr.texture_move"
    bl_label = "Move Texture"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(
        items=[
            ('UP', "Up", ""),
            ('DOWN', "Down", ""),
        ]
    )

    def execute(self, context):
        settings = context.object.utr_settings
        index = settings.active_texture_index
        target = index - 1 if self.direction == 'UP' else index + 1

        if not 0 <= target < len(settings.textures):
            return {'CANCELLED'}

        settings.textures.move(index, target)
        settings.active_texture_index = target
        return {'FINISHED'}


class UTR_OT_texture_get_from_materials(bpy.types.Operator):
    bl_idname = "utr.texture_get_from_materials"
    bl_label = "Get from Materials"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.object
        settings = obj.utr_settings
        images = core.collect_material_images(obj)

        if not images:
            self.report({'WARNING'}, "No image textures found in materials")
            return {'CANCELLED'}

        existing = {
            item.image.as_pointer()
            for item in settings.textures
            if item.image
        }

        added = 0
        for image in images:
            pointer = image.as_pointer()
            if pointer in existing:
                continue

            item = settings.textures.add()
            item.image = image
            existing.add(pointer)
            added += 1

        if not added:
            self.report({'INFO'}, "All material images are already listed")
            return {'CANCELLED'}

        settings.active_texture_index = len(settings.textures) - 1
        self.report({'INFO'}, f"Added {added} image(s)")
        return {'FINISHED'}


class UTR_OT_texture_add(bpy.types.Operator):
    bl_idname = "utr.texture_add"
    bl_label = "Add Texture"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(context.object and context.object.type == 'MESH')

    def execute(self, context):
        settings = context.object.utr_settings
        settings.textures.add()
        settings.active_texture_index = len(settings.textures) - 1
        return {'FINISHED'}


class UTR_OT_texture_remove(bpy.types.Operator):
    bl_idname = "utr.texture_remove"
    bl_label = "Remove Texture"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return bool(obj and obj.type == 'MESH' and obj.utr_settings.textures)

    def execute(self, context):
        settings = context.object.utr_settings
        index = settings.active_texture_index
        settings.textures.remove(index)
        settings.active_texture_index = max(0, min(index, len(settings.textures) - 1))
        return {'FINISHED'}


class UTR_OT_reproject(bpy.types.Operator):
    bl_idname = "utr.reproject"
    bl_label = "Reproject"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return bool(obj and obj.type == 'MESH' and obj.mode == 'OBJECT')

    def execute(self, context):
        obj = context.object
        settings = obj.utr_settings

        valid, message = core.validate_settings(obj)
        if not valid:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        outputs = []

        try:
            triangles = core.build_uv_triangles(obj)

            for item in settings.textures:
                output = core.reproject_image(item.image, triangles, settings)
                output.use_fake_user = True
                outputs.append(output)

        except Exception as exc:
            for output in outputs:
                bpy.data.images.remove(output, do_unlink=True)

            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        self.report({'INFO'}, f"Created {len(outputs)} texture(s)")
        return {'FINISHED'}


CLASSES = (
    UTR_OT_texture_add,
    UTR_OT_texture_remove,
    UTR_OT_texture_move,
    UTR_OT_texture_get_from_materials,
    UTR_OT_reproject,
)