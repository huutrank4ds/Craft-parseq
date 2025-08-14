import torch.hub
import cv2
import numpy as np

def resize_img(img, target_size=(32, 128), pad=0):
    target_h, target_w = target_size
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # Tính vị trí để đặt ảnh vào giữa
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    padded = np.full((target_h, target_w, 3), pad, dtype=np.uint8)
    padded[top:top + new_h, left:left + new_w] = resized
    return padded

def get_transform(img_list, target_size=(32, 128)):
    batch_tensor = []
    for img in img_list:
        resized_img = resize_img(img, target_size)
        batch_tensor.append(resized_img)
    return torch.stack(batch_tensor, dim=0)

def get_recognizer(device):
    return torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(device)
