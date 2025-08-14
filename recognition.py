import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T

def get_transform(img_size=(32, 128)):
    transforms = []
    transforms.extend([
        T.Resize(img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])
    return T.Compose(transforms)

def get_recognizer(device):
    return torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(device)
