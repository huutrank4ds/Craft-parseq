from detection import get_detector, test_net
from logging import getLogger
from utils import preprocess_img
import torch

LOGGER = getLogger(__name__)

class NaHOCR():
    def __init__(self, pretrained, device='cpu', det_model_path=None,
                 detector=True, recognizer=False, 
                 verbose=True, quantize=True, cudnn_benchmark=False, refine=None):
        
        self.pretrained = pretrained
        self.device = device
        self.quantize = quantize
        self.cudnn_benchmark = cudnn_benchmark
        self.det_model_path = det_model_path
        self.verbose = verbose
        self.refine = refine

        if self.device == 'cpu':
            if self.verbose:
                LOGGER.warning("Running on CPU, performance may be slow.")
        elif self.device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Please check your PyTorch installation or use CPU.")
        else:
            if self.verbose:
                LOGGER.warning("Unknown device type. Defaulting to CPU.")
            self.device = 'cpu'

        if detector:
            detector_path = self.getDetectorPath()
            self.detector = self.initDetector(detector_path)

        if recognizer:
            self.initRecognizer()

    def initDetector(self, detector_path):
        return self.get_detector(detector_path, self.device, self.quantize, self.cudnn_benchmark)

    def getDetectorPath(self):
        self.get_textbox = test_net
        self.get_detector = get_detector
        return self.det_model_path if self.det_model_path else 'pretrained/craft_mlt_25k.pth'
    
    def initRecognizer(self):
        pass


    def detect(self, img, canvas_size=1280, text_threshold=0.4, 
               link_threshold=0.7, low_text=0.2, mag_ratio=1.5, preprocess=True):
        if preprocess:
            img = preprocess_img(img)
        boxes, polys, ret_scores =  self.get_textbox(net=self.detector, image=img, canvas_size=canvas_size, mag_ratio=mag_ratio,
                                                      text_threshold=text_threshold, link_threshold=link_threshold, 
                                                        low_text=low_text, poly=False, device=self.device, refine_net=self.refine)
        return boxes, polys

    def recognize(self, img, preprocess=True):
        if preprocess:
            img = preprocess_img(img)
        # Call the recognizer model here
        return
