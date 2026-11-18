
import bpy
from . import core
from bpy.props import EnumProperty, IntProperty
from time import perf_counter


_UTR_STATUS = {
    "active": False,
    "progress": 0.0,
    "cancel": False,
    "label": "Reprojecting...",
}


def _draw_utr_statusbar(header, context):
    layout = header.layout

    layout.template_input_status()
    layout.separator_spacer()
    layout.template_reports_banner()
    layout.template_running_jobs()

    if _UTR_STATUS["active"]:
        row = layout.row(align=True)
        row.label(text="", icon='RENDER_STILL')
        row.label(text=_UTR_STATUS["label"])

        progress = row.row()
        progress.scale_x = 2.0
        progress.progress(
            factor=_UTR_STATUS["progress"],
            text=f"{int(_UTR_STATUS['progress'] * 100)}%",
            type='BAR',
        )

        row.operator_context = 'INVOKE_REGION_WIN'
        row.operator(
            "utr.cancel_reproject",
            text="",
            icon='PANEL_CLOSE',
        )

    layout.separator_spacer()
    row = layout.row()
    row.alignment = 'RIGHT'
    layout.template_status_info()


def _print_profile(profile, total):
    print("\n[UTR] Reprojection Profile")
    print(f"Geometry: {profile['geometry']:.3f}s")

    for (w, h), timings in profile["maps"]:
        raster = timings["raster_details"]

        if timings["raster_backend"] == "GPU":
            raster_text = (
                f"Binning {raster['binning']:.3f}s | "
                f"Upload {raster['upload']:.3f}s | "
                f"Dispatch {raster['dispatch']:.3f}s"
            )
        else:
            raster_text = f"CPU {timings['raster']:.3f}s"

        print(
            f"Map {w}x{h}: "
            f"Raster ({timings['raster_backend']}) {raster_text} | "
            f"JFA {timings['jfa']:.3f}s ({timings['jfa_backend']}) | "
            f"Map Upload {timings['map_upload']:.3f}s"
        )

        if timings.get("fallback_reason"):
            print(f"Fallback: {timings['fallback_reason']}")

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


class UTR_OT_cancel_reproject(bpy.types.Operator):
    bl_idname = "utr.cancel_reproject"
    bl_label = "Cancel Reprojection"
    bl_description = "Stop texture reprojection"

    @classmethod
    def poll(cls, context):
        return _UTR_STATUS["active"] and not _UTR_STATUS["cancel"]

    def execute(self, context):
        _UTR_STATUS["cancel"] = True
        _UTR_STATUS["label"] = "Canceling..."
        context.workspace.status_text_set_internal(None)
        return {'FINISHED'}


class UTR_OT_reproject(bpy.types.Operator):
    bl_idname = "utr.reproject"
    bl_label = "Reproject"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return bool(
            obj and obj.type == 'MESH'
            and obj.mode == 'OBJECT'
            and not _UTR_STATUS["active"]
        )

    def invoke(self, context, event):
        obj = context.object
        settings = obj.utr_settings

        valid, message = core.validate_settings(obj)
        if not valid:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        groups = {}
        for item in settings.textures:
            key = tuple(map(int, item.image.size))
            groups.setdefault(key, []).append(item.image)

        self._obj = obj
        self._settings = settings
        self._wm = context.window_manager
        self._workspace = context.workspace
        self._timer = None
        self._cleaned = False

        self._groups = list(groups.items())
        self._group_index = 0
        self._image_index = 0
        self._images = None
        self._mapping = None
        self._triangles = None
        self._phase = "GEOMETRY"

        self._outputs = []
        self._completed = 0
        self._total_steps = 1 + len(self._groups) + len(settings.textures)
        self._start_time = perf_counter()

        self._profile = {
            "geometry": 0.0,
            "maps": [],
            "images": [],
        }

        _UTR_STATUS.update({
            "active": True,
            "progress": 0.0,
            "cancel": False,
            "label": "Reprojecting...",
        })

        self._workspace.status_text_set(_draw_utr_statusbar)
        self._workspace.status_text_set_internal(None)

        self._timer = self._wm.event_timer_add(
            0.05, window=context.window
        )
        self._wm.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    def _update_progress(self):
        self._completed += 1
        _UTR_STATUS["progress"] = min(
            1.0, self._completed / self._total_steps
        )
        self._workspace.status_text_set_internal(None)

    def _process_step(self):
        if self._phase == "GEOMETRY":
            start = perf_counter()
            self._triangles = core.build_uv_triangles(self._obj)
            self._profile["geometry"] = perf_counter() - start

            self._phase = "MAP"
            self._update_progress()
            return False

        if self._phase == "MAP":
            if self._group_index >= len(self._groups):
                return True

            (w, h), self._images = self._groups[self._group_index]

            mapping, timings = core.build_reprojection_map(
                self._triangles, w, h, self._settings
            )

            self._profile["maps"].append(((w, h), timings))
            self._mapping = mapping
            self._image_index = 0
            self._phase = "IMAGE"

            self._update_progress()
            return False

        if self._phase == "IMAGE":
            image = self._images[self._image_index]

            result, timings = core.reproject_image(
                image, self._mapping, self._settings
            )

            result.use_fake_user = True
            self._outputs.append(result)
            self._profile["images"].append((image.name, timings))

            self._image_index += 1
            self._update_progress()

            if self._image_index >= len(self._images):
                self._mapping = None
                self._images = None
                self._group_index += 1
                self._phase = "MAP"

            return self._completed >= self._total_steps

        return False

    def _cleanup(self):
        if self._cleaned:
            return

        self._cleaned = True

        if self._timer is not None:
            self._wm.event_timer_remove(self._timer)
            self._timer = None

        try:
            self._workspace.status_text_set(None)
        except ReferenceError:
            pass

        _UTR_STATUS.update({
            "active": False,
            "progress": 0.0,
            "cancel": False,
            "label": "Reprojecting...",
        })

    def _finish(self, cancelled=False, error=None):
        total = perf_counter() - self._start_time
        self._cleanup()

        if cancelled or error is not None:
            for output in self._outputs:
                if output.name in bpy.data.images:
                    bpy.data.images.remove(output, do_unlink=True)

        _print_profile(self._profile, total)

        if cancelled:
            self.report({'WARNING'}, "Reprojection cancelled")
            return {'CANCELLED'}

        if error is not None:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}

        self.report(
            {'INFO'},
            f"Created {len(self._outputs)} textures in {total:.2f}s"
        )
        return {'FINISHED'}

    def modal(self, context, event):
        if event.type == 'ESC':
            return self._finish(cancelled=True)

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if _UTR_STATUS["cancel"]:
            return self._finish(cancelled=True)

        try:
            if self._process_step():
                return self._finish()
        except Exception as exc:
            return self._finish(error=exc)

        return {'RUNNING_MODAL'}

    def cancel(self, context):
        self._cleanup()


CLASSES = (
    UTR_OT_texture_add,
    UTR_OT_texture_remove,
    UTR_OT_texture_move,
    UTR_OT_texture_get_from_materials,
    UTR_OT_update_materials,
    UTR_OT_cancel_reproject,
    UTR_OT_reproject,
)