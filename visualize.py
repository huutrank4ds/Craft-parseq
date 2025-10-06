import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def overlay_heatmap(original_image, score_map, alpha=0.7, beta=0.3, gamma=0):
    """
    Đè bản đồ nhiệt region và link lên ảnh gốc một cách chính xác.
    Hàm này xử lý việc chuyển đổi heatmap 1 kênh thành ảnh màu 3 kênh.

    Args:
        original_image (np.array): Ảnh gốc BGR. Phải là uint8.
        score_map (np.array): Bản đồ điểm từ CRAFT, shape (H, W, 2).
                               Kênh 0 là region, kênh 1 là link.
        alpha (float): Trọng số của ảnh gốc.
        beta (float): Trọng số của bản đồ nhiệt.
        gamma (float): Giá trị vô hướng được cộng vào tổng.

    Returns:
        A tuple containing two images: (region_overlay, link_overlay).
    """
    # Đảm bảo ảnh gốc là uint8
    if original_image.dtype != np.uint8:
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.clip(0, 255).astype(np.uint8)
    
    # Lấy kích thước của ảnh hiển thị (đã resize)
    h, w, _ = original_image.shape

    # 1. Tách riêng region và link map từ score_map
    region_map = score_map[:, :, 0] # Shape: (H_map, W_map)
    link_map = score_map[:, :, 1]   # Shape: (H_map, W_map)

    # Hàm phụ để tạo một overlay duy nhất
    def _create_single_overlay(heatmap_2d):
        # Resize heatmap về kích thước của ảnh hiển thị
        heatmap_resized = cv2.resize(heatmap_2d, (w, h))
        
        # Chuẩn hóa giá trị về 0-255 và chuyển kiểu thành uint8
        heatmap_8bit = (heatmap_resized * 255).astype(np.uint8)
        
        # *** BƯỚC QUAN TRỌNG NHẤT: ÁP DỤNG BẢN ĐỒ MÀU ***
        # Biến ảnh xám 1 kênh thành ảnh màu giả 3 kênh
        heatmap_colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
        
        # Bây giờ cả hai ảnh đều có shape (h, w, 3) và có thể trộn được
        overlay_image = cv2.addWeighted(original_image, alpha, heatmap_colored, beta, gamma)
        return overlay_image

    # Tạo cả hai ảnh overlay
    region_overlay = _create_single_overlay(region_map)
    link_overlay = _create_single_overlay(link_map)

    return (region_overlay, link_overlay)
    

def draw_polygons(image, polygons, color=(0, 255, 0), thickness=2):
    """
    Vẽ một danh sách các đa giác (polygons) lên trên một ảnh.

    Args:
        image (np.array): Ảnh đầu vào (đọc bằng OpenCV, định dạng BGR).
        polygons (list): Danh sách các đa giác cho ảnh này.
                         Mỗi đa giác là một mảng NumPy các điểm.
        color (tuple): Màu của đường viền, theo định dạng BGR.
                       Mặc định là màu xanh lá (0, 255, 0).
        thickness (int): Độ dày của đường viền.

    Returns:
        np.array: Một bản sao của ảnh đầu vào với các kết quả đã được vẽ lên.
    """
    # 1. Tạo một bản sao của ảnh để không làm thay đổi ảnh gốc
    result_image = image.copy()

    # 2. Lặp qua tất cả các đa giác trong danh sách
    for poly in polygons:
        # Bỏ qua nếu đa giác không hợp lệ (mặc dù nó đã được xử lý trong test_net)
        if poly is None:
            continue

        # 3. Chuẩn bị dữ liệu cho OpenCV
        # Chuyển đổi các điểm thành một mảng NumPy với kiểu dữ liệu int32
        pts = np.array(poly, dtype=np.int32)
        # Reshape lại để có định dạng (Số_điểm, 1, 2) mà cv2.polylines yêu cầu
        pts = pts.reshape((-1, 1, 2))

        # 4. Vẽ đa giác lên ảnh
        cv2.polylines(result_image, [pts], isClosed=True, color=color, thickness=thickness)

    return result_image
