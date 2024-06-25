# docres.py

import torch
import cv2
import numpy as np
from enum import Enum
from api import NodeContext
from nodes.properties.inputs import ImageInput, TextInput
from nodes.properties.outputs import ImageOutput
from nodes.impl.pytorch.utils import np2tensor, tensor2np
from nodes.utils.utils import get_h_w_c

from .inference_gradio import inference_one_image, model_init

from ...settings import PyTorchSettings, get_settings
from .. import processing_group

MODEL_PATH = "C:/chainner/resources/src/packages/chaiNNer_pytorch/pytorch/processing/checkpoints/docres.pkl"
possible_tasks = [
    "dewarping",
    "deshadowing",
    "appearance",
    "deblurring",
    "binarization",
]

class TaskType(Enum):
    DEWARPING = "dewarping"
    DESHADOWING = "deshadowing"
    APPEARANCE = "appearance"
    DEBLURRING = "deblurring"
    BINARIZATION = "binarization"

def run_tasks(image: np.ndarray, tasks: str) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # CPU for large images by RAHUL
    #device = "cpu"

    # Load model
    model = model_init(MODEL_PATH, device)

    # Parse tasks
    task_list = tasks.split(",")
    for task in task_list:
        task = task.strip()
        if task not in possible_tasks:
            raise ValueError(f"Invalid task: {task}")

    # Convert image to the required format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Run inference
    bgr_image = image[..., ::-1].copy()
    bgr_restored_image = inference_one_image(model, bgr_image, task_list, device)
    if bgr_restored_image.ndim == 3:
        rgb_image = bgr_restored_image[..., ::-1]
    else:
        rgb_image = bgr_restored_image

    return rgb_image

@processing_group.register(
    schema_id="chainner:pytorch:docres",
    name="DocRes",
    description=[
        "Apply selected document restoration tasks such as dewarping, deshadowing, etc."
    ],
    icon="MdCropFree",
    inputs=[
        ImageInput(label="Image", channels=None),  # Accept any number of channels
        TextInput(label="Tasks", default="appearance"),
    ],
    outputs=[
        ImageOutput().with_never_reason("Returns the enhanced image after applying the selected tasks.")
    ],
    node_context=True,
)
def docres_node(context: NodeContext, target_img: np.ndarray, tasks: str) -> np.ndarray:
    # Apply document restoration tasks
    result_img = run_tasks(target_img, tasks)
    return result_img
