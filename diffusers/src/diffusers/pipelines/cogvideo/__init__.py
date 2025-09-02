from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["pipeline_cogvideox"] = ["CogVideoXPipeline"]
    _import_structure["pipeline_cogvideox_image2video"] = ["CogVideoXImageToVideoPipeline"]
    _import_structure["pipeline_cogvideox_video2video"] = ["CogVideoXVideoToVideoPipeline"]
    _import_structure["pipeline_cogvideox_inpainting"] = ["CogVideoXInpaintPipeline"]
    _import_structure["pipeline_cogvideox_inpainting_branch"] = ["CogVideoXDualInpaintPipeline"]
    _import_structure["pipeline_cogvideox_inpainting_sft"] = ["CogVideoXSFTInpaintPipeline"]
    _import_structure["pipeline_cogvideox_inpainting_selfguidance"] = ["CogVideoXSelfGuidanceInpaintPipeline"]
    _import_structure["pipeline_cogvideox_image2video_inpainting"] = ["CogVideoXImageToVideoInpaintPipeline"]
    _import_structure["pipeline_cogvideox_inpainting_i2v_branch3"] = ["CogVideoXI2VTriInpaintPipeline"]
    _import_structure["pipeline_cogvideox_inpainting_i2v_branch3_sep"] = ["CogVideoXI2VTriInpaintPipeline_sep"]
    _import_structure["pipeline_cogvideox_inpainting_i2v_branch3_112"] = ["CogVideoXI2VTriInpaint112Pipeline"]
    _import_structure["pipeline_cogvideox_inpainting_i2v_branch_anyl"] = ["CogVideoXI2VDualInpaintAnyLPipeline"]
    _import_structure["pipeline_cogvideox_inpainting_i2v_anyl"] = ["CogVideoXI2VInpaintAnyLPipeline"]
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        from .pipeline_cogvideox import CogVideoXPipeline
        from .pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
        from .pipeline_cogvideox_video2video import CogVideoXVideoToVideoPipeline
        from .pipeline_cogvideox_inpainting import CogVideoXInpaintPipeline
        from .pipeline_cogvideox_inpainting_branch import CogVideoXDualInpaintPipeline
        from .pipeline_cogvideox_inpainting_selfguidance import CogVideoXSelfGuidanceInpaintPipeline
        from .pipeline_cogvideox_inpainting_sft import CogVideoXSFTInpaintPipeline
        from .pipeline_cogvideox_image2video_inpainting import CogVideoXImageToVideoInpaintPipeline
        from .pipeline_cogvideox_inpainting_i2v_branch3 import CogVideoXI2VTriInpaintPipeline
        from .pipeline_cogvideox_inpainting_i2v_branch3_112 import CogVideoXI2VTriInpaint112Pipeline
        from .pipeline_cogvideox_inpainting_i2v_branch3_sep import CogVideoXI2VTriInpaintPipeline_sep
        from .pipeline_cogvideox_inpainting_i2v_anyl import CogVideoXI2VInpaintAnyLPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
