# Skew Correction Node for ChaiNNer

import cv2
import numpy as np
import logging
from api import NodeContext
from nodes.properties.inputs import ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput
from nodes.impl.pytorch.utils import np2tensor, tensor2np
from nodes.utils.utils import get_h_w_c
from ...settings import PyTorchSettings, get_settings
from .. import processing_group

# Configure logging
logging.basicConfig(level=logging.INFO)

def _ensure_gray(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        pass
    return image

def _ensure_optimal_square(image):
    assert image is not None, image
    nw = nh = cv2.getOptimalDFTSize(max(image.shape[:2]))
    output_image = cv2.copyMakeBorder(
        src=image,
        top=0,
        bottom=nh - image.shape[0],
        left=0,
        right=nw - image.shape[1],
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )
    return output_image

def _get_fft_magnitude(image):
    gray = _ensure_gray(image)
    opt_gray = _ensure_optimal_square(gray)

    # Ensure the image is uint8
    if opt_gray.dtype != np.uint8:
        opt_gray = opt_gray.astype(np.uint8)

    # thresh
    opt_gray = cv2.adaptiveThreshold(
        ~opt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
    )

    # perform fft
    dft = np.fft.fft2(opt_gray)
    shifted_dft = np.fft.fftshift(dft)

    # get the magnitude (module)
    magnitude = np.abs(shifted_dft)
    return magnitude

def _get_angle_radial_projection(m, angle_max=None, num=None, W=None):
    """Get angle via radial projection."""

    assert m.shape[0] == m.shape[1]
    r = c = m.shape[0] // 2

    if angle_max is None:
        pass

    if num is None:
        num = 20

    tr = np.linspace(-1 * angle_max, angle_max, int(angle_max * num * 2)) / 180 * np.pi
    profile_arr = tr.copy()

    def f(t):
        _f = np.vectorize(
            lambda x: m[c + int(x * np.cos(t)), c + int(-1 * x * np.sin(t))]
        )
        _l = _f(range(0, r))
        val_init = np.sum(_l)
        return val_init

    vf = np.vectorize(f)
    li = vf(profile_arr)

    a = tr[np.argmax(li)] / np.pi * 180

    if a == -1 * angle_max:
        return 0
    return a

def get_skewed_angle(image: np.ndarray, vertical_image_shape: int = None, angle_max: float = None):
    """Getting angle from a given document image."""

    assert isinstance(image, np.ndarray), image

    if angle_max is None:
        angle_max = 15

    # resize
    if vertical_image_shape is not None:
        ratio = vertical_image_shape / image.shape[0]
        image = cv2.resize(image, None, fx=ratio, fy=ratio)

    m = _get_fft_magnitude(image)
    a = _get_angle_radial_projection(m, angle_max=angle_max)
    return a

def correct_text_skewness(image, angle_max=15):
    """
    Method to rotate image by n degree
    @param image:
    return:
    """
    h, w = image.shape[:2]
    x_center, y_center = (w // 2, h // 2)

    # Find angle to rotate image
    rotation_angle = get_skewed_angle(image, angle_max=angle_max)
    logging.info(f"[INFO]: Rotation angle is {rotation_angle}")

    # Rotate the image by given n degree around the center of the image
    M = cv2.getRotationMatrix2D((x_center, y_center), rotation_angle, 1.0)
    borderValue = (255, 255, 255)

    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=borderValue)
    return rotated_image

@processing_group.register(
    schema_id="chainner:opencv:skew_correction",
    name="Skew Correction",
    description=[
        "Correct skewness in the input image using FFT-based skew detection and correction."
    ],
    icon="MdCropFree",
    inputs=[
        ImageInput(label="Image", channels=None),  # Accept any number of channels
        SliderInput(
            "Max Angle",
            min=1,
            max=45,
            default=15,
            precision=0,
            step=1,
            unit="deg",
        )
        .with_docs(
            "Maximum angle to search for skewness correction.",
            "Higher values allow for detecting larger skew angles.",
            hint=True,
        ),
    ],
    outputs=[
        ImageOutput().with_never_reason("Returns the skew-corrected image.")
    ],
    node_context=True,
)
def skew_correction_node(context, target_img: np.ndarray, max_angle: float) -> np.ndarray:
    # Log input image details
    #logging.info(f"Input image shape: {target_img.shape}")
    #logging.info(f"Input image dtype: {target_img.dtype}")

    # Ensure image is in uint8 format
    if target_img.dtype != np.uint8:
        target_img = (target_img * 255).astype(np.uint8)

    # Apply skew correction
    result_img = correct_text_skewness(target_img, angle_max=max_angle)

    # Convert back to original format if necessary
    if target_img.dtype == np.float32:
        result_img = result_img.astype(np.float32) / 255.0

    # Log output image details
    #logging.info(f"Output image shape: {result_img.shape}")
    #logging.info(f"Output image dtype: {result_img.dtype}")

    return result_img

# Installation instructions (same as provided in your example)
