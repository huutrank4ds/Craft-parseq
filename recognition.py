import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T

class ImgTransform():
    def __init__(self, img_size=(32, 128)):
        self.img_size = img_size
        self.transform = self._get_transform()

    def _get_transform(self):
        transforms = []
        transforms.extend([
            T.Resize(self.img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        return T.Compose(transforms)
    
    def __call__(self, imgs):
        if isinstance(imgs, list) or isinstance(imgs, np.ndarray):
            if isinstance(imgs[0], np.ndarray):
                imgs = [Image.fromarray(img) for img in imgs]
        elif isinstance(imgs, Image.Image):
            imgs = [imgs]
        else:
            raise ValueError("Unsupported image format. Please provide a list of images or numpy arrays.")
        if isinstance(imgs, np.ndarray):
            imgs = [Image.fromarray(imgs)]
        batch_tensors = []
        for img in imgs:
            batch_tensors.append(self.transform(img))
        return torch.stack(batch_tensors, dim=0)
    

def get_recognizer(device, parallel=True):
    model_rec = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True)
    model_rec.eval()
    model_rec = model_rec.to(device)
    if parallel:
        model_rec = torch.nn.DataParallel(model_rec)
    return model_rec
