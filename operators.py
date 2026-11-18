
import bpy
from . import core
from bpy.props import EnumProperty, IntProperty
from time import perf_counter


def _print_profile(profile, total):
    print("\n[UTR] Reprojection Profile")
    print(f"Geometry: {profile['geometry']:.3f}s")

    for (w, h), timings in profile["maps"]:
        jfa_label = "JFA submit" if timings.get("jfa_async") else "JFA"

        print(
            f"Map {w}x{h}: "
            f"Raster {timings['raster']:.3f}s | "
            f"{jfa_label} {timings['jfa']:.3f}s "
            f"({timings['backend']}) | "
            f"Upload {timings['map_upload']:.3f}s"
        )

    for name, timings in profile["images"]:
        labels = {
            "read": "Read",
            "upload": "Upload",
            "sample": "Dispatch",
            "readback": "Readback/Sync",
            "map_readback": "Map readback",
            "create": "Create",
            "write": "Write",
        }

        parts = [
            f"{labels[key]} {timings[key]:.3f}s"
            for key in labels
            if key in timings
        ]

        print(f"{name}: {' | '.join(parts)} ({timings['backend']})")

        if timings.get("fallback_reason"):
            print(f"Fallback: {timings['fallback_reason']}")

    print(f"Total: {total:.3f}s\n")


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


class UTR_OT_update_materials(bpy.types.Operator):
    bl_idname = "utr.update_materials"
    bl_label = "Update Materials"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(context.object and context.object.type == 'MESH')

    def execute(self, context):
        obj = context.object

        try:
            updated, missing = core.update_materials(
                obj, obj.utr_settings
            )
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        if not updated:
            message = (
                "No matching texture nodes found"
                if not missing else
                "No reprojected textures found"
            )
            self.report({'WARNING'}, message)
            return {'CANCELLED'}

        message = f"Updated {updated} texture nodes"

        if missing:
            message += f"; {len(missing)} output(s) missing"
            self.report({'WARNING'}, message)
        else:
            self.report({'INFO'}, message)

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

        groups = {}
        for item in settings.textures:
            if item.image:
                key = tuple(map(int, item.image.size))
                groups.setdefault(key, []).append(item.image)

        total_steps = 1 + len(groups) + len(settings.textures)
        progress = 0
        wm = context.window_manager

        total_start = perf_counter()
        profile = {
            "geometry": 0.0,
            "maps": [],
            "images": [],
        }
        outputs = []

        wm.progress_begin(0, total_steps)

        try:
            start = perf_counter()
            triangles = core.build_uv_triangles(obj)
            profile["geometry"] = perf_counter() - start

            progress += 1
            wm.progress_update(progress)

            for (w, h), images in groups.items():
                mapping, timings = core.build_reprojection_map(
                    triangles, w, h, settings
                )
                profile["maps"].append(((w, h), timings))

                progress += 1
                wm.progress_update(progress)

                for image in images:
                    result, timings = core.reproject_image(
                        image, mapping, settings
                    )
                    result.use_fake_user = True
                    outputs.append(result)
                    profile["images"].append((image.name, timings))

                    progress += 1
                    wm.progress_update(progress)

        except Exception as exc:
            for output in outputs:
                bpy.data.images.remove(output, do_unlink=True)

            total = perf_counter() - total_start
            _print_profile(profile, total)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        finally:
            wm.progress_end()

        total = perf_counter() - total_start
        _print_profile(profile, total)

        self.report(
            {'INFO'},
            f"Created {len(outputs)} textures in {total:.2f}s"
        )
        return {'FINISHED'}


CLASSES = (
    UTR_OT_texture_add,
    UTR_OT_texture_remove,
    UTR_OT_texture_move,
    UTR_OT_texture_get_from_materials,
    UTR_OT_update_materials,
    UTR_OT_reproject,
)