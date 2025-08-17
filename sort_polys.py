import numpy as np
from sklearn.cluster import DBSCAN

def polygon_to_rect(poly):
    """
    Chuyển đổi một polygon (danh sách các đỉnh) thành một hình chữ nhật bao
    quanh đơn giản (x_min, y_min, x_max, y_max).
    """
    x_coords = [p[0] for p in poly]
    y_coords = [p[1] for p in poly]
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

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

def calculate_min_edge_distance(box1_rect, box2_rect):
    """
    Tính khoảng cách tối thiểu giữa các cạnh của hai hình chữ nhật.
    Nếu hai box chồng chéo, khoảng cách trả về là 0.
    """
    x1_min, y1_min, x1_max, y1_max = box1_rect
    x2_min, y2_min, x2_max, y2_max = box2_rect

    # Khoảng cách ngang (horizontal gap)
    delta_x = max(0, x1_min - x2_max, x2_min - x1_max)
    
    # Khoảng cách dọc (vertical gap)
    delta_y = max(0, y1_min - y2_max, y2_min - y1_max)
    
    # Khoảng cách Euclidean của các khoảng trống
    return np.sqrt(delta_x**2 + delta_y**2)

def sort_by_robust_cluster_and_reading_order(items_to_sort, eps=20, min_samples=1):
    """
    Sắp xếp bounding box bằng phương pháp MẠNH MẼ:
    1. Xây dựng ma trận khoảng cách dựa trên KHOẢNG CÁCH TỐI THIỂU GIỮA CÁC CẠNH.
    2. Phân cụm bằng DBSCAN với ma trận khoảng cách này.
    3. Sắp xếp các cụm, sau đó sắp xếp bên trong mỗi cụm.

    Args:
        items_to_sort (list): Danh sách các tuple (polygon, data).
        eps (float): Tham số 'epsilon' cho DBSCAN. Quan trọng: bây giờ nó đại diện
                     cho KHOẢNG TRỐNG (WHITESPACE) tối đa cho phép giữa hai box
                     để chúng được coi là cùng một cụm.
        min_samples (int): Số lượng box tối thiểu để tạo thành một cụm.

    Returns:
        list: Danh sách các tuple đã được sắp xếp hoàn chỉnh.
    """
    if not items_to_sort:
        return []

    num_items = len(items_to_sort)
    
    # 1. Chuyển đổi tất cả polygon sang dạng hình chữ nhật (rect) để tính toán
    rects = [polygon_to_rect(item[0]) for item in items_to_sort]

    # 2. Xây dựng Ma trận Khoảng cách Tối thiểu giữa các Cạnh
    distance_matrix = np.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(i, num_items):
            dist = calculate_min_edge_distance(rects[i], rects[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist # Ma trận đối xứng

    # 3. Chạy DBSCAN với ma trận khoảng cách đã tính toán
    # metric='precomputed' là tham số quan trọng nhất ở đây!
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distance_matrix)
    labels = db.labels_

    # 4. Gom nhóm các item gốc dựa trên nhãn (label) của cụm
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(items_to_sort[i])

    # 5. Sắp xếp các cụm từ trên-xuống, trái-sang-phải
    cluster_ids = [k for k in clusters if k != -1]
    def get_cluster_start_position(cluster_id):
        # Lấy tọa độ y_min và x_min của toàn bộ cụm
        cluster_rects = [polygon_to_rect(item[0]) for item in clusters[cluster_id]]
        min_y = min(rect[1] for rect in cluster_rects)
        min_x = min(rect[0] for rect in cluster_rects)
        return (min_y, min_x)
    cluster_ids.sort(key=get_cluster_start_position)

    # 6. Sắp xếp bên trong mỗi cụm và kết hợp kết quả
    final_sorted_list = []
    for cid in cluster_ids:
        cluster_items = clusters[cid]
        sorted_cluster = sort_reading_order(cluster_items) # Gọi hàm con
        final_sorted_list.extend(sorted_cluster)

    # 7. Xử lý nhiễu (nếu có)
    if -1 in clusters:
        noise_items = clusters[-1]
        sorted_noise = sort_reading_order(noise_items)
        final_sorted_list.extend(sorted_noise)

    return final_sorted_list