# Pipeline OCR end-to-end (NaHOCR) sử dụng CRAFT và Parseq

[![Python](https://img.shields.io/badge/Python-3.x-informational)](https://www.python.org/)
![Status](https://img.shields.io/badge/status-demo-success)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

**Pipeline OCR end-to-end** dùng nhận dạng văn bản bên trong hình ảnh sử dụng **CRAFT** làm detector và **Parseq** làm recognizer.

## Mục lục
- [✨ Tính năng](#-tính-năng)
- [📦 Cài đặt & Chạy](#-cài-đặt--chạy)

---

## ✨ Tính năng
- Nhận diện văn bản bên trong hình ảnh.
- Kiến trúc nhẹ cho tốc độ nhanh mà vẫn có độ chính xác cao.
- Mạnh với ảnh chứa văn bản tiếng Anh nhưng vẫn chính xác trên cả văn bản có dấu như tiếng Việt.
- Pipeline end-to-end dễ sử dụng.

---

## 📦 Cài đặt & Chạy
Clone repo:
```bash
git clone https://github.com/huutrank4ds/Craft-parseq.git
cd Craft-parseq
pip install -r requirements.txt
```
Sử dụng mô hình:
  ```bash
from nahocr import NaHOCR
# Sử dụng gpu để inference
pipe = NaHOCR(device='cuda')
# Sử dụng cpu để inference, không truyền mặc định mô hình dùng cpu
# pipe = NaHOCR(device='cpu')

# Tạo ảnh giả
import cv2
import numpy as np

height = 300
width = 300
image = np.ones((height, width, 3), dtype=np.uint8) * 255
cv2.putText(image, 'Test OCR', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(image, 'NaHOCR', (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
cv2.putText(image, 'HuHu HaHa', (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

# Chạy pipeline ocr, đầu vào phải là batch ảnh (numpy arrray, đường dẫn)
from detection_utils import draw_polygons
import matplotlib.pyplot as plt

results_ocr = pipe.ocr([image])
plt.imshow(draw_polygons(image, results_ocr[0]['bboxes']))
print(' '.join(results_ocr[0]['texts']))
  ```
Các tham số khác:
| Tham số    | Kiểu dữ liệu         | Nội dung     |
|----------------|------------------|----------------|
| det_model_path  | str             | Đường dẫn tới thư file chứa trọng số khác của mô hình CRAFT, mặc định None mô hình dùng pretrained mặc định|
| detector     | bool    | Mặc định True, mô hình có sử dụng detector      |
| recognizer       | bool  | Mặc định True, mô hình có sử dụng recognizer |
| verbose | bool | Mặc định True, có xuất thêm log chi tiết |
| parallel | bool | Mặc định True, có thực hiện chạy song song đa gpu |
| refine | str | Mặc định None, đường dẫn đến trọng số mô hình RefineNet |
| quantize | bool | Mặc định True, giảm kích thước mô hình khi inference, đánh đổi độ chính xác nhưng không đáng kể |
| cudnn_benchmark | bool | Mặc định True, tối ưu tốc độ khi chạy trên gpu (tốt khi kích thước input cố định) |

Tham số của method ocr:
- batch_size_det: Độ lớn batch khi detect
- batch_size_rec: Độ lớn batch khi recognize
- custom_process: Hàm xử lý ảnh cần tạo ra 2 danh sách, ảnh xử lý trước khi detect và ảnh xử lý trước khi recognize
- custom_setting_det: Dict chứa tham số cho detect
```bash
# Mặc định setting cho detect
setting_detect = {
    'canvas_size': 1280,
    'text_threshold': 0.4,
    'link_threshold': 0.7,
    'low_text': 0.2,
    'mag_ratio': 1.5,
    'preprocess': True
}
```
- Chỉ dùng detector:
```bash
# Đầu ra là batch danh sách các bouding box cho từng từ
results_det = pipe.detect(image)
plt.imshow(draw_polygons(image, results_det[0]))
```
- Chỉ dùng recognizer:
```bash
results_rec = pipe.recognize(image)
```




