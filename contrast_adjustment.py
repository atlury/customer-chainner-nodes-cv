# Contrast Adjustment Node for ChaiNNer

import cv2
import numpy as np
from api import NodeContext
from nodes.properties.inputs import ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput
from nodes.impl.pytorch.utils import np2tensor, tensor2np
from nodes.utils.utils import get_h_w_c
from ...settings import PyTorchSettings, get_settings
from .. import processing_group

def adjust_contrast(image: np.ndarray, contrast_coeff: float) -> np.ndarray:
    # Convert image to float
    float_image = image / 255.0

    # Compute mean values over each channel
    mean = np.mean(float_image, axis=(0, 1), keepdims=True)

    # Adjust contrast
    new_image = (float_image - mean) * contrast_coeff + mean

    # Clip values to ensure they fall between 0 and 1
    new_image = np.clip(new_image, a_min=0., a_max=1.)

    # Convert back to uint8
    new_image = (new_image * 255).astype(np.uint8)

    return new_image

@processing_group.register(
    schema_id="chainner:opencv:contrast_adjustment",
    name="Contrast Adjustment",
    description=[
        "Adjust contrast of the input image using a specified contrast coefficient."
    ],
    icon="MdBrightness6",
    inputs=[
        ImageInput(label="Image", channels=None),  # Accept any number of channels
        SliderInput(
            "Contrast Coefficient",
            min=0.5,
            max=2.0,
            default=1.0,
            precision=2,
            step=0.1,
            unit="",
        )
        .with_docs(
            "Coefficient to adjust the contrast of the image.",
            "Higher values increase contrast, while lower values decrease it.",
            hint=True,
        ),
    ],
    outputs=[
        ImageOutput().with_never_reason("Returns the contrast-adjusted image.")
    ],
    node_context=True,
)
def contrast_adjustment_node(context, target_img: np.ndarray, contrast_coeff: float) -> np.ndarray:
    # Ensure image is in uint8 format
    if target_img.dtype != np.uint8:
        target_img = (target_img * 255).astype(np.uint8)

    # Apply contrast adjustment
    result_img = adjust_contrast(target_img, contrast_coeff)

    # Convert back to original format if necessary
    if target_img.dtype == np.float32:
        result_img = result_img.astype(np.float32) / 255.0

    return result_img

# Installation instructions (same as provided in your example)
