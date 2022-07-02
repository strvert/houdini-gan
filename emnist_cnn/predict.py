# This code assumes the node has the following parameters:
#     res: an integer vector of size 2 with default (640, 480)
#     stillimage: a toggle with default on
#     start: an integer with default 1
#     length: an integer with default 240

import array
import numpy as np
import inlinecpp
import io
from typing import Dict
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision
from PIL import Image

AVAILABLE_GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if AVAILABLE_GPU else 'cpu')


def load_mapping(alltext: str) -> Dict[int, str]:
    lines = alltext.split('\n')
    return {int(pair[0]): int(pair[1]) for pair in [line.strip().split(' ') for line in lines]}


class CNNNetMinimum(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(16 * 15 * 15, 120)
        self.fc_out = nn.Linear(120, 47)

        self.relu = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(-1, 16 * 15 * 15)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x

    @staticmethod
    def network_name():
        return "cnn-minimum"


def resolution(cop_node):
    return [1, 1]


def required_input_planes(cop_node, output_plane):
    return ("0", "A")


def output_planes_to_cook(cop_node):
    return ("A")


def cook(cop_node, plane, resolution):
    if plane == "A":
        input_cop = cop_node.inputs()[0]
        trained_model = cop_node.parm("model").eval()
        mapping_file = cop_node.parm("mapping_txt").eval()

        cnn_net: nn.Module = CNNNetMinimum().to(DEVICE)
        file_buffer = io.BytesIO(hou.readBinaryFile(trained_model))
        cnn_net.load_state_dict(torch.load(file_buffer))

        mapping_text = hou.readFile(mapping_file)
        class_mapping = load_mapping(mapping_text)

        plane_ndarray = np.frombuffer(input_cop.allPixelsAsString(plane), dtype=np.float16);
        plane_tensor = torch.from_numpy(plane_ndarray.astype(np.float32)).to(DEVICE)

        output = cnn_net(plane_tensor.reshape(1, 28, 28))
        result = torch.argmax(output[0])
        print(class_mapping[result.item()])

        cop_node.setPixelsOfCookingPlane([np.random.rand()])
