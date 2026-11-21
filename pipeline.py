from . import sampling


from .mapping import ensure_cpu_mapping
from .models import ReprojectionError, UVOverlapError


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

    result, timings = sampling._reproject_image_cpu(
        image, mapping, settings
    )
    timings["map_readback"] = map_readback

    if settings.compute_backend == 'AUTO' and mapping.gpu_error:
        timings["backend"] = "CPU fallback"
        timings["fallback_reason"] = mapping.gpu_error
    else:
        timings["backend"] = "CPU"

    return result, timings