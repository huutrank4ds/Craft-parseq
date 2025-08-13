import cv2
import numpy as np

def overlay_heatmap(original_image, full_heatmap,
                          heatmap_type='region', alpha=0.7, beta=0.3, gamma=0):
    """
    Đè một hoặc nhiều bản đồ nhiệt lên ảnh gốc với các tùy chọn linh hoạt.

    Args:
        original_image (np.array): Ảnh gốc BGR hoặc RGB.
        full_heatmap (np.array): Ảnh bản đồ nhiệt đầy đủ (chứa cả region và link).
        heatmap_type (str): Loại bản đồ nhiệt để tạo.
                            Chấp nhận: 'region', 'link', 'both'.
        alpha (float): Trọng số của ảnh gốc.
        beta (float): Trọng số của bản đồ nhiệt.
        gamma (float): Giá trị vô hướng được cộng vào tổng.

    Returns:
        - Nếu heatmap_type là 'region' hoặc 'link': Trả về một ảnh (np.array) đã được trộn.
        - Nếu heatmap_type là 'both': Trả về một tuple chứa hai ảnh: (region_overlay, link_overlay).
    """
    # Lấy kích thước của ảnh gốc
    h, w, _ = original_image.shape
    
    # Hàm phụ để tạo một ảnh overlay đơn lẻ
    def _create_single_overlay(heatmap_part):
        heatmap_resized = cv2.resize(heatmap_part, (w, h))
        return cv2.addWeighted(original_image, alpha, heatmap_resized, beta, gamma)

    heatmap_width = full_heatmap.shape[1]
    
    if heatmap_type == 'region':
        # Chỉ lấy nửa bên trái và trả về một ảnh
        score_text_heatmap = full_heatmap[:, :heatmap_width // 2]
        return _create_single_overlay(score_text_heatmap)
    
    elif heatmap_type == 'link':
        # Chỉ lấy nửa bên phải và trả về một ảnh
        score_link_heatmap = full_heatmap[:, heatmap_width // 2:]
        return _create_single_overlay(score_link_heatmap)
        
    elif heatmap_type == 'both':
        # Tạo hai ảnh overlay riêng biệt và trả về dưới dạng tuple
        score_text_heatmap = full_heatmap[:, :heatmap_width // 2]
        score_link_heatmap = full_heatmap[:, heatmap_width // 2:]
        
        region_overlay = _create_single_overlay(score_text_heatmap)
        link_overlay = _create_single_overlay(score_link_heatmap)
        
        return (region_overlay, link_overlay)
    
    else:
        raise ValueError("heatmap_type must be 'region', 'link', or 'both'")
    

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