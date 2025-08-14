import torch.hub
import cv2
import numpy as np

import torch
import numpy as np
import cv2

class ImageTransform:
    """
    Một lớp để tiền xử lý một danh sách các ảnh NumPy thành một batch tensor.

    Các bước xử lý bao gồm:
    1. Chuẩn hóa kênh màu (đảm bảo ảnh là RGB).
    2. Thay đổi kích thước ảnh về kích thước mục tiêu trong khi vẫn giữ nguyên tỉ lệ,
       sau đó đệm (pad) phần còn thiếu.
    3. Chuyển đổi batch ảnh NumPy thành một batch tensor PyTorch và chuẩn hóa giá trị
       về khoảng [-1.0, 1.0].
    """
    def __init__(self, img_size=(32, 128)):
        """
            Khởi tạo transform.
            Args:
                img_size (tuple): Kích thước mục tiêu (height, width).
        """
        self.target_h, self.target_w = img_size

    def _normalize_channels(self, img):
        """
            Đảm bảo ảnh đầu ra luôn là ảnh 3 kênh RGB.
            - Chuyển ảnh grayscale (1 kênh) thành RGB.
            - Chuyển ảnh RGBA (4 kênh) thành RGB bằng cách bỏ kênh alpha.
        """
        if img.ndim == 2:
            # Ảnh là grayscale, chuyển sang RGB
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            # Ảnh là RGBA, chuyển sang RGB
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img

    def _resize_and_pad(self, img):
        """
            Thay đổi kích thước ảnh để vừa với kích thước mục tiêu mà vẫn giữ tỉ lệ,
            sau đó đệm phần còn lại để đạt kích thước mục tiêu.
        """
        h, w = img.shape[:2]
        
        # Tính tỉ lệ resize sao cho chiều dài nhất vừa với kích thước mục tiêu
        scale = min(self.target_w / w, self.target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Tạo một ảnh nền (padded) với kích thước mục tiêu và màu đen (giá trị 0)
        padded_img = np.full((self.target_h, self.target_w, 3), 0, dtype=np.uint8)
        
        # Tính toán vị trí để đặt ảnh đã resize vào giữa ảnh nền
        top = (self.target_h - new_h) // 2
        left = (self.target_w - new_w) // 2
        
        # Đặt ảnh vào
        padded_img[top:top + new_h, left:left + new_w] = resized_img
        
        return padded_img

    def _finalize_batch(self, np_images):
        """
            Chuyển một danh sách các ảnh NumPy đã xử lý thành một batch tensor cuối cùng.
        """
        if not np_images:
            return torch.empty(0, 3, self.target_h, self.target_w)

        # Xếp chồng các ảnh NumPy thành một batch duy nhất
        batch_np = np.stack(np_images, axis=0)
        
        # Chuyển đổi sang tensor, đổi chiều (N,H,W,C -> N,C,H,W) và chuẩn hóa
        batch_tensor = torch.from_numpy(batch_np)
        batch_tensor = batch_tensor.permute(0, 3, 1, 2).float() / 255.0
        batch_tensor = (batch_tensor - 0.5) / 0.5
        
        return batch_tensor

    def __call__(self, img_list: list) -> torch.Tensor:
        """
        Thực thi toàn bộ pipeline tiền xử lý.
        
        Args:
            img_list (list): Một danh sách các ảnh NumPy.
            
        Returns:
            torch.Tensor: Một batch tensor có dạng (N, C, H, W).
        """
        processed_images = []
        for img in img_list:
            # Bước 1: Chuẩn hóa kênh màu
            img_rgb = self._normalize_channels(img)
            # Bước 2: Thay đổi kích thước và đệm
            img_padded = self._resize_and_pad(img_rgb)
            processed_images.append(img_padded)
        
        # Bước 3: Hoàn thiện batch (chuyển sang tensor và chuẩn hóa)
        final_batch = self._finalize_batch(processed_images)
        
        return final_batch

def get_recognizer(device):
    return torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(device)
