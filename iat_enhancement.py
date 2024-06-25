# iat_enhancement.py

import os
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, ConvertImageDtype
from api import NodeContext
from nodes.properties.inputs import ImageInput, EnumInput
from nodes.properties.outputs import ImageOutput
from nodes.impl.pytorch.utils import tensor2np
from nodes.utils.utils import get_h_w_c
from enum import Enum
from ...settings import PyTorchSettings, get_settings
from .. import processing_group
from .iatenhancement.model.IAT import IAT  # Ensure the correct import path

# Define the model paths using absolute paths
MODEL_PATH_DARK = os.path.join(os.path.dirname(__file__), 'iatenhancement/checkpoint/best_Epoch_lol.pth')
MODEL_PATH_EXPOSURE = os.path.join(os.path.dirname(__file__), 'iatenhancement/checkpoint/best_Epoch_exposure.pth')

class EnhancementType(Enum):
    DARK = "Dark"
    EXPOSURE = "Exposure"

def tensor_to_numpy(tensor):
    tensor = tensor.detach().cpu().numpy()
    if tensor.ndim == 3 and tensor.shape[0] == 3:  # Convert CHW to HWC
        tensor = tensor.transpose(1, 2, 0)
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)  # Ensure the output is uint8
    return tensor

def inference(img, enhancement_type):
    model = IAT()
    if enhancement_type == "dark":
        checkpoint_file_path = MODEL_PATH_DARK
    else:
        checkpoint_file_path = MODEL_PATH_EXPOSURE

    state_dict = torch.load(checkpoint_file_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    transform = Compose([
        ToTensor(), 
        Resize(384),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        ConvertImageDtype(torch.float)
    ])
    input_img = transform(img)

    with torch.no_grad():
        enhanced_img = model(input_img.unsqueeze(0))

    result_img = tensor_to_numpy(enhanced_img[0])
    return result_img

@processing_group.register(
    schema_id="chainner:custom:iat_enhancement",
    name="IAT Enhancement",
    description=[
        "Apply IAT model for low-light enhancement or exposure correction."
    ],
    icon="MdPhotoFilter",
    inputs=[
        ImageInput(label="Image", channels=None),  # Accept any number of channels
        EnumInput(
            EnhancementType,
            label="Enhancement Type",
            default=EnhancementType.DARK,
            option_labels={
                EnhancementType.DARK: "Low-light Enhancement",
                EnhancementType.EXPOSURE: "Exposure Correction",
            },
        ),
    ],
    outputs=[
        ImageOutput().with_never_reason("Returns the enhanced image.")
    ],
    node_context=True,
)
def iat_enhancement_node(context: NodeContext, target_img: np.ndarray, enhancement_type: EnhancementType) -> np.ndarray:
    result_img = inference(target_img, enhancement_type.value.lower())
    return result_img

# Installation instructions (same as provided in your example)
