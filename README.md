# UV Textures Reprojection

**UV Textures Reprojection** is a Blender addon for transferring image textures from one UV layout to another.

It builds a mapping between a source UV map and a destination UV map, then creates reprojected image copies at the same resolution as the source textures. A dedicated progress bar is displayed during processing, and the operation can be cancelled.

## Features

- Reprojects textures between two UV maps on the same mesh
- Supports CPU and GPU compute backends
- Offers nearest-neighbor, linear, and cubic interpolation
- Processes multiple textures, including textures with different resolutions
- Detects overlapping destination triangles
- Provides image-boundary or fixed-radius dilation
- Automatically finds image texture nodes in the object's materials
- Replaces matching image nodes and switches them to the destination UV map
- Shows processing progress and backend timing information

## Requirements

- Blender 5.3 or newer
- A mesh object with at least two different UV maps
- Object Mode to run reprojection
- For GPU processing, a Blender-compatible GPU compute backend

The addon requires NumPy, which is available in Blender's bundled Python environment.

## Installation

1. Download or clone this repository.
2. In Blender, open **Edit > Preferences**.
3. Open the **Extensions** or **Add-ons** section.
4. Install the addon package or select this repository folder if installing from source.
5. Enable **UV Textures Reprojection**.

## Quick Start

1. Select a mesh object.
2. Open **Properties > Material**.
3. Under **UV Textures Reprojection**, select:
   - **Source UV**: the UV layout currently used by the texture
   - **Destination UV**: the UV layout that should receive the texture
4. Add one or more textures to the texture list.
5. Configure interpolation, compute backend, dilation, and overlap behavior.
6. Click **Reproject**.

The addon creates one output image for each selected source texture. Processing runs in the background with a progress indicator in the Blender status bar. Press **Esc** or use the close button in the status bar to cancel.

## Texture Selection

Use the `+` button to add an empty texture slot, or use the texture options menu:

- **Get from Materials** collects image textures from all material nodes on the active mesh.
- **Update Materials** replaces matching source textures in material nodes with the most recent reprojected output.

Textures can be reordered with the arrow buttons.

## Settings

| Setting | Description |
|---|---|
| Source UV | UV map used to read the current texture layout. |
| Destination UV | UV map where the texture will be rebuilt. |
| Interpolation | Pixel sampling method: nearest-neighbor, linear, or cubic. |
| Compute Backend | Use Auto, CPU, or GPU compute. |
| Dilation | Image Bounds fills outside the UV islands to the image edges; Fixed Padding expands only by the configured radius. |
| Padding | Pixel radius used by Fixed Padding. |
| Overlaps | Abort on destination UV overlap or overwrite overlapping areas. |
| Output Suffix | Text appended to every generated image name. |

### Compute Backend

- **Auto** tries GPU compute and falls back to CPU if a GPU stage fails.
- **CPU** uses the CPU implementation only.
- **GPU** requires successful GPU compute and reports an error if a required GPU stage is unavailable.

GPU processing is not used for sampling if the current map cannot be prepared successfully.

### Dilation

**Image Bounds** uses the nearest seeded UV position for pixels outside the destination UV islands, filling the available image area.

**Fixed Padding** limits dilation to a circular pixel radius defined by **Padding**.

Pixels outside the selected destination UV coverage remain transparent or are initialized with zero alpha.

### Destination UV Overlaps

If two destination triangles cover the same pixel, the addon can either:

- **Abort** reprojection before creating the output, or
- **Overwrite** the earlier result using the later triangle.

## Material Update

After reprojection, use **Update Materials** to:

1. Replace matching source image nodes with the generated output images.
2. Reuse or create a `UV Map` node.
3. Connect that node to updated image texture nodes.
4. Set the UV Map node to the configured destination UV map.

The operation processes node trees used by the active object's material slots and nested node groups.

## Generated Images

Each output image is named:

```text
<source image name><output suffix>
```

The default suffix is `_reprojected`.

Generated images store Blender custom properties identifying the source image and output suffix. This allows **Update Materials** to find the matching reprojection result for a source texture.

## Limitations

- UV coordinates must use the 0–1 range.
- Source UV and destination UV must be different named UV maps.
- The active object must be a mesh in Object Mode.
- File and generated images are supported.
- Packed or regular generated Blender images can be processed.
- Multiview images are not supported.
- A texture's output resolution matches the texture image dimensions.
- GPU-specific failures are reported and may trigger the configured CPU fallback.

## Development

The addon is implemented as a Blender package with separate modules for:

- `properties.py`: addon settings and property groups
- `panel.py`: the Material Properties interface
- `operators.py`: texture, reprojection, cancellation, and material operators
- `core.py`: UV triangulation, mapping, CPU sampling, validation, and material updates
- `gpu_backend.py`: GPU rasterization, jump-flood mapping, sampling, and readback

The Blender console prints a timing profile after completion or cancellation, including geometry, rasterization, jump-flood, image readback, and image creation timings.
