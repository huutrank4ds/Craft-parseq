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

def batch_transform(imgs, img_size=(32, 128)):
    if isinstance(imgs, list) or isinstance(imgs, np.ndarray):
        if isinstance(imgs[0], np.ndarray):
            imgs = [Image.fromarray(img) for img in imgs]
    elif isinstance(imgs, Image.Image):
        imgs = [imgs]
    if isinstance(imgs, np.ndarray):
        imgs = [Image.fromarray(imgs)]
    else:
        raise ValueError("Unsupported image format. Please provide a list of images or numpy arrays.")
    batch_tensors = []
    img_transform = get_transform(img_size)
    for img in imgs:
        batch_tensors.append(img_transform(img))
    return torch.stack(batch_tensors, dim=0)

def get_recognizer(device):
    return torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True).eval().to(device)
