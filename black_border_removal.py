# Black Border Removal Node for ChaiNNer

import cv2
import numpy as np
from enum import Enum
from api import NodeContext
from nodes.properties.inputs import ImageInput, EnumInput, SliderInput
from nodes.properties.outputs import ImageOutput
from nodes.impl.pytorch.utils import np2tensor, tensor2np
from nodes.utils.utils import get_h_w_c
from ...settings import PyTorchSettings, get_settings
from .. import processing_group

class ImageType(Enum):
    RGB = "RGB"
    GRAYSCALE = "Grayscale"

def remove_black_border(image: np.ndarray, sensitivity: float, image_type: ImageType) -> np.ndarray:
    if not (0 < sensitivity <= 1):
        raise ValueError("Sensitivity must be between 0 and 1.")

    if image_type == ImageType.GRAYSCALE:
        if len(image.shape) != 2:
            raise ValueError("Expected a grayscale image but got an image with multiple channels.")
        img = image.T
    else:
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image but got a grayscale or non-RGB image.")
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).T  # Convert to grayscale for processing

    w, h = img.shape
    mean_of_img = np.mean(img) * sensitivity

    if mean_of_img == 0:
        raise ValueError("Mean of image is zero, invalid sensitivity value.")

    dataw = [w, 0]
    datah = [h, 0]

    for i in range(w):
        if np.mean(img[i]) > mean_of_img:
            if i < dataw[0]:
                dataw[0] = i
            else:
                dataw[1] = i

    img = img.T

    for i in range(h):
        if np.mean(img[i]) > mean_of_img:
            if i < datah[0]:
                datah[0] = i
            else:
                datah[1] = i

    cropped_image = image[datah[0]:datah[1], dataw[0]:dataw[1]]

    return cropped_image

@processing_group.register(
    schema_id="chainner:opencv:black_border_removal",
    name="Black Border Removal",
    description=[
        "Remove black borders from the input image using sensitivity to determine the border threshold."
    ],
    icon="MdCropFree",
    inputs=[
        ImageInput(label="Image", channels=None),  # Accept any number of channels
        EnumInput(
            ImageType,
            label="Image Type",
            default=ImageType.RGB,
            option_labels={
                ImageType.RGB: "RGB",
                ImageType.GRAYSCALE: "Grayscale",
            },
        ),
        SliderInput(
            "Sensitivity",
            min=0.1,
            max=1.0,
            default=0.5,
            precision=2,
            step=0.1,
            unit="",
        )
        .with_docs(
            "Sensitivity to determine the border threshold.",
            "Higher values result in more aggressive border removal.",
            hint=True,
        ),
    ],
    outputs=[
        ImageOutput().with_never_reason("Returns the image with black borders removed.")
    ],
    node_context=True,
)
def black_border_removal_node(context, target_img: np.ndarray, image_type: ImageType, sensitivity: float) -> np.ndarray:
    # Apply black border removal
    result_img = remove_black_border(target_img, sensitivity, image_type)

    return result_img

# Installation instructions (same as provided in your example)
