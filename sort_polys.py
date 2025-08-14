import numpy as np

def calculate_box_properties(box):
    """
        Tính toán các thuộc tính cần thiết của một bounding box.
        
        Args:
            box (list of lists): Một đa giác, ví dụ [[x1, y1], [x2, y2], ...].
            
        Returns:
            tuple: (tâm x, tâm y, chiều cao, chiều rộng, tọa độ y trên cùng).
    """
    # Chuyển đổi sang mảng NumPy để tính toán dễ dàng
    poly = np.array(box, dtype="float32")
    
    # Tọa độ x và y
    x_coords = poly[:, 0]
    y_coords = poly[:, 1]
    
    # Tính toán tâm và kích thước
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    height = np.max(y_coords) - np.min(y_coords)
    width = np.max(x_coords) - np.min(x_coords)
    
    # Lấy tọa độ y trên cùng để tìm box bắt đầu mỗi dòng
    top_y = np.min(y_coords)
    
    return center_x, center_y, height, width
    
def sort_reading_order(items_to_sort, vertical_tolerance=0.5):
    """
        Sắp xếp các box theo thứ tự đọc từ trên xuống dưới, trái sang phải một cách linh hoạt.

        Hàm này nhóm các box thành các dòng dựa trên sự gần gũi theo chiều dọc của tâm,
        sau đó sắp xếp các box trong mỗi dòng từ trái sang phải.

        Args:
            items_to_sort (list): Danh sách các tuple (box, image).
            vertical_tolerance (float): Hệ số xác định mức độ linh hoạt khi nhóm các box
                                        vào cùng một dòng. Giá trị lớn hơn cho phép độ
                                        nghiêng của dòng lớn hơn. 0.7 là một giá trị tốt.

        Returns:
            list: Danh sách các tuple (box, image) đã được sắp xếp.
    """
    if not items_to_sort:
        return []

    # 1. Chuẩn bị: Gắn các thuộc tính tính toán vào mỗi item để không phải tính lại
    # item_with_props = [(item, (cx, cy, h, w, top_y)), ...]
    items_with_props = [
        (item, calculate_box_properties(item[0])) for item in items_to_sort
    ]

    sorted_items = []
    
    # 2. Vòng lặp chính: Tiếp tục cho đến khi tất cả các item được xử lý
    while items_with_props:
        # 2a. Chọn box "hạt giống": Tìm box có tọa độ y trung tâm nhỏ nhất (cao nhất trên trang)
        seed_item_with_props = min(items_with_props, key=lambda x: x[1][1]) # Sắp xếp theo cy
        
        # Lấy thông tin của box hạt giống
        (_, (seed_cx, seed_cy, seed_h, _)) = seed_item_with_props
        
        # 2b. Tìm tất cả các box nằm trên cùng một dòng với box hạt giống
        current_line = []
        remaining_items = []
        
        for item_prop in items_with_props:
            item_data, (item_cx, item_cy, item_h, _) = item_prop
            
            # Điều kiện để xác định có cùng dòng không:
            # Khoảng cách dọc giữa hai tâm phải nhỏ hơn một ngưỡng linh hoạt.
            # Ngưỡng này là chiều cao trung bình của hai box nhân với hệ số tolerance.
            threshold = ((seed_h + item_h) / 2.0) * vertical_tolerance
            
            if abs(seed_cy - item_cy) < threshold:
                current_line.append(item_prop)
            else:
                remaining_items.append(item_prop)
        
        # 2c. Sắp xếp dòng hiện tại từ trái sang phải dựa vào tâm x
        current_line.sort(key=lambda x: x[1][0]) # Sắp xếp theo center_x
        
        # 2d. Thêm các item của dòng đã sắp xếp (chỉ lấy lại dữ liệu gốc) vào kết quả cuối cùng
        sorted_items.extend([item for item, props in current_line])
        
        # 2e. Cập nhật danh sách các item còn lại để xử lý cho các dòng tiếp theo
        items_with_props = remaining_items
            
    return sorted_items