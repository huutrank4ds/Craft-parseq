import cv2
from imgproc import loadImage
import numpy as np

def preprocess_img(img_input):
    """
    Xử lý các định dạng đầu vào khác nhau để trả về một batch ảnh.

    Args:
        img_input: Đầu vào có thể là:
            - str: Đường dẫn đến một tệp ảnh.
            - np.ndarray: Một ảnh (H, W, C) hoặc một batch ảnh (N, H, W, C).
            - list: Một danh sách các đường dẫn (str) hoặc các ảnh (np.ndarray).

    Returns:
        np.ndarray: Một batch ảnh có dạng (N, H, W, C).
    
    Raises:
        TypeError: Nếu định dạng đầu vào không được hỗ trợ.
        ValueError: Nếu mảng NumPy có số chiều không hợp lệ hoặc danh sách chứa các mục không đồng nhất.
    """
    # Trường hợp 1: Đầu vào là một đường dẫn (string)
    if isinstance(img_input, str):
        # Tải ảnh và bọc nó trong một mảng NumPy để tạo thành một batch có 1 ảnh
        loaded_img = loadImage(img_input)
        return np.expand_dims(loaded_img, axis=0)

    # Trường hợp 2: Đầu vào là một mảng NumPy
    elif isinstance(img_input, np.ndarray):
        # Nếu mảng có 4 chiều (N, H, W, C), nó đã là một batch
        if len(img_input.shape) == 4:
            return img_input
        # Nếu mảng có 3 chiều (H, W, C), nó là một ảnh đơn
        elif len(img_input.shape) == 3:
            # Thêm một chiều ở đầu để tạo thành một batch có 1 ảnh
            return np.expand_dims(img_input, axis=0)
        else:
            raise ValueError(f"Mảng NumPy đầu vào có số chiều không hợp lệ: {len(img_input.shape)}. Chỉ chấp nhận 3 hoặc 4 chiều.")

    # Trường hợp 3: Đầu vào là một danh sách
    elif isinstance(img_input, list):
        if not img_input:
            return np.array([]) # Trả về một batch rỗng nếu danh sách rỗng
        
        processed_list = []
        for item in img_input:
            if isinstance(item, str):
                processed_list.append(loadImage(item))
            elif isinstance(item, np.ndarray) and len(item.shape) == 3:
                processed_list.append(item)
            else:
                raise TypeError(f"Loại dữ liệu trong danh sách không được hỗ trợ: {type(item)}")
        
        # Chuyển danh sách các ảnh thành một batch NumPy duy nhất
        return np.array(processed_list)

    # Trường hợp khác: Ném ra lỗi
    else:
        raise TypeError(f"Định dạng đầu vào không được hỗ trợ: {type(img_input)}")
