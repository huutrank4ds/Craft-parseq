from detection import get_detector, test_net
from recognition import get_recognizer, get_transform, batch_transform
from logging import getLogger
from utils import preprocess_img
import torch
import os
import numpy as np
import cv2
from sort_polys import sort_reading_order
from PIL import Image

LOGGER = getLogger(__name__)

device_available = {'cpu', 'cuda'}

class NaHOCR():
    def __init__(self, pretrained=True, device='cpu', det_model_path=None,
                 detector=True, recognizer=True, 
                 verbose=True, quantize=True, cudnn_benchmark=False, refine=None):
        
        self.pretrained = pretrained
        self.quantize = quantize
        self.cudnn_benchmark = cudnn_benchmark
        self.det_model_path = det_model_path
        self.verbose = verbose
        self.refine = refine

        if device == 'cpu':
            if self.verbose:
                LOGGER.warning("Running on CPU, performance may be slow.")
            self.device = torch.device('cpu')
        elif device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Please check your PyTorch installation or use CPU.")
            self.device = torch.device('cuda')
        else:
            if self.verbose:
                LOGGER.warning("Unknown device type. Defaulting to CPU.")
            self.device = torch.device('cpu')

        if detector:
            detector_path = self.getDetectorPath()
            self.detector = self.initDetector(detector_path)

        if recognizer:
            self.recognizer = self.initRecognizer()

    def initDetector(self, detector_path):
        return self.get_detector(detector_path, self.device, self.quantize, self.cudnn_benchmark)

    def getDetectorPath(self):
        self.get_textbox = test_net
        self.get_detector = get_detector
        if self.det_model_path:
            return self.det_model_path
        else:
            # Lấy thư mục hiện tại của file này
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'pretrained/craft_mlt_25k.pth')
            return model_path
    
    def initRecognizer(self):
        self.transform = batch_transform(img_size=(32, 128))
        return get_recognizer(self.device)


    def detect(self, img, canvas_size=1280, text_threshold=0.4, 
               link_threshold=0.7, low_text=0.2, mag_ratio=1.5, preprocess=True):
        """
            Phát hiện văn bản trong ảnh.
            Args:
                - img: Ảnh hoặc đường dẫn cục bộ hoặc một batch ảnh (đường dẫn)
                - canvas_size, mag_ratio: Kích thước và tỷ lệ phóng đại của ảnh
                - preprocess: Có thực hiện tiền xử lý ảnh hay không
            Returns:
                - polys: Các polygon chứa văn bản phát hiện được 
        """
        if preprocess:
            img = preprocess_img(img)
        boxes, polys, ret_scores =  self.get_textbox(net=self.detector, image=img, canvas_size=canvas_size, mag_ratio=mag_ratio,
                                                      text_threshold=text_threshold, link_threshold=link_threshold, 
                                                        low_text=low_text, poly=False, device=self.device, refine_net=self.refine)
        return polys

    def recognize(self, imgs, transform=True):
        if transform:
            imgs = self.transform(imgs)
        imgs = imgs.to(self.device)
        with torch.no_grad():
            logits = self.recognizer(imgs)
        decoded_labels, confidences = self.recognizer.tokenizer.decode(logits.softmax(-1))
        return decoded_labels, confidences
    
    def orderPoints(self, pts):
        """
            Sắp xếp 4 điểm tọa độ theo thứ tự:
            trên-trái, trên-phải, dưới-phải, dưới-trái
        
            Args:
                - pts: list chứa 4 điểm
            Return:
                - List 4 điểm đã sắp xếp
        """
        # Khởi tạo một danh sách tọa độ và sắp xếp chúng
        # dựa trên tổng (x+y) và hiệu (y-x)
        rect = np.zeros((4, 2), dtype="float32")
        if not isinstance(pts, np.ndarray):
            pts = np.array(pts)
        # Điểm trên-trái sẽ có tổng x+y nhỏ nhất
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        # Điểm dưới-phải sẽ có tổng x+y lớn nhất
        rect[2] = pts[np.argmax(s)]
        # Điểm trên-phải sẽ có hiệu y-x nhỏ nhất
        # Điểm dưới-trái sẽ có hiệu y-x lớn nhất
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    def cropAndWarp(self, image, four_points):
        """
            Cắt và duỗi thẳng vùng tứ giác từ ảnh.
        
            Args:
                - image: Ảnh đầu vào (đọc bằng OpenCV).
                - four_points: Một list hoặc numpy array chứa 4 điểm, ví dụ: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            Return:
                - Ảnh đã được cắt và duỗi thẳng.
        """
        # Chuyển four_points thành numpy array
        pts = np.array(four_points, dtype="float32")
    
        # Sắp xếp các điểm theo thứ tự chuẩn
        rect = self.orderPoints(pts)
        (tl, tr, br, bl) = rect
    
        # Tính chiều rộng của ảnh đầu ra
        # là khoảng cách lớn nhất giữa điểm dưới-phải và dưới-trái
        # hoặc trên-phải và trên-trái
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
    
        # Tính chiều cao của ảnh đầu ra
        # là khoảng cách lớn nhất giữa điểm trên-phải và dưới-phải
        # hoặc trên-trái và dưới-trái
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Nếu chiều rộng hoặc chiều cao quá nhỏ, trả về None để tránh lỗi
        if maxWidth == 0 or maxHeight == 0:
            return None
    
        # Tạo các điểm đích cho ảnh đầu ra (một hình chữ nhật hoàn hảo)
        dst = np.array([[0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype="float32")
    
        # Tính toán ma trận biến đổi phối cảnh M
        M = cv2.getPerspectiveTransform(rect, dst)
    
        # Áp dụng ma trận M để cắt và duỗi thẳng ảnh
        warped_image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped_image
    
    def getImageList(self, polys, img_rgb, sort_output = True):
        """
            Trích xuất các vùng ảnh từ một danh sách các đa giác (polygons).
        
            Args:
                polys (list): Danh sách các đa giác. Mỗi đa giác là một danh sách các điểm,
                            ví dụ: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
                img_rgb (np.ndarray): Ảnh gốc ở định dạng RGB.
                sort_output (bool): Nếu True, sắp xếp các ảnh được trích xuất theo thứ tự đọc.
        
            Returns:
                list: Một danh sách các tuple, mỗi tuple chứa (box, crop_img).
        """
        image_list = []
        
        # Duyệt qua danh sách các đa giác (polys)
        for box in polys:
            # Chuyển đổi box thành mảng NumPy để dùng cho hàm warp
            rect = np.array(box, dtype="float32")
            
            # Sử dụng hàm crop và warp để trích xuất ảnh từ đa giác
            crop_img = self.cropAndWarp(img_rgb, rect)
            
            # Chỉ thêm vào danh sách nếu trích xuất thành công
            if crop_img is not None:
                image_list.append((box, crop_img))
                
        # Sắp xếp kết quả đầu ra theo thứ tự đọc nếu được yêu cầu
        if sort_output:
            image_list = sort_reading_order(image_list)
            
        return image_list

    def ocr(self, imgs, batch_det_size=1, batch_rec_size=1, custom_process=None):
        """
            Thực hiện OCR trên một danh sách các ảnh.
        
            Args:
                imgs (list): Danh sách các ảnh hoặc đường dẫn đến ảnh.
        
            Returns:
                list: Danh sách các tuple chứa (box, crop_img, text, confidence).
        """
        # Kiểm tra kiểu dữ liệu đầu vào và đọc path ảnh
        if isinstance(imgs, list) or isinstance(imgs, np.ndarray):
            if isinstance(imgs[0], str):
                rgb_imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in imgs]
                # Xử lý ảnh không đọc được
                valid_ids = [i for i, img in enumerate(rgb_imgs) if img is not None]
                valid_rgb_imgs = [rgb_imgs[i] for i in valid_ids]
                valid_img_paths = [imgs[i] for i in valid_ids]
            elif isinstance(imgs[0], np.ndarray):
                valid_ids = [i for i, img in enumerate(imgs) if img is not None]
                valid_rgb_imgs = [imgs[i] for i in valid_ids]
                valid_img_paths = ['' for _ in valid_ids]
            elif isinstance(imgs[0], Image.Image):
                valid_ids = [i for i, img in enumerate(imgs) if img is not None]
                valid_rgb_imgs = [np.array(img.convert('RGB')) for img in imgs if img is not None]
                valid_img_paths = ['' for _ in valid_ids]
            else:
                raise ValueError("Unsupported image format. Please provide a list of image paths or numpy arrays.")
            
        if custom_process:
            valid_rgb_imgs = custom_process(valid_rgb_imgs)
        # Thực hiện detect văn bản
        text_boxes = self.detect(valid_rgb_imgs)

        all_patches_info = []
        for idx, img_path in enumerate(valid_img_paths):
            image_patches = self.getImageList(text_boxes[idx], valid_rgb_imgs[idx])
            
            for box, cropped_img in image_patches:
                # Lưu lại thông tin để map kết quả về sau
                all_patches_info.append({
                    'original_image_idx': idx,
                    'box': box,
                    'patch': cropped_img
                })
        # Thực hiện recognize cho tất cả các patch
        all_rec_texts = []
        all_confs = []

        all_patches = [patch_info['patch'] for patch_info in all_patches_info]

        for i in range(0, len(all_patches), batch_rec_size):
            # Tạo batch các patch ảnh
            batch_of_patches = all_patches[i:i + batch_rec_size]
            # Nhận dạng batch hiện tại
            batch_texts, batch_confs = self.recognize(batch_of_patches)
            # Thêm kết quả của batch vào danh sách tổng
            all_rec_texts.extend(batch_texts)
            all_confs.extend(batch_confs)

        # Phân phối kết quả
        final_results = [
            {
                "input_path": path,
                "bboxes": [],
                "texts": [],
                "confidences": []
            } for path in valid_img_paths
        ]
        for i, info in enumerate(all_patches_info):
            original_idx = info['original_image_idx']
            final_results[original_idx]['bboxes'].append(info['box'])
            final_results[original_idx]['texts'].append(all_rec_texts[i])
            final_results[original_idx]['confidences'].append(all_confs[i])

        return final_results
