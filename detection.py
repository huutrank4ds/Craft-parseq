import imgproc
import torch
import cv2
import craft_utils
import numpy as np
from craft import CRAFT
from load_weights import copyStateDict
import torch.backends.cudnn as cudnn

def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device, refine_net=None):

    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:                                                        # image is single numpy array
        image_arrs = [image]

    img_resized_list = []
    # resize
    for img in image_arrs:
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(img, canvas_size,
                                                                      interpolation=cv2.INTER_LINEAR,
                                                                      mag_ratio=mag_ratio)
        img_resized_list.append(img_resized)
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    x = [np.transpose(imgproc.normalizeMeanVariance(n_img), (2, 0, 1))
         for n_img in img_resized_list]
    x = torch.from_numpy(np.array(x))
    x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    score_text_batch = y[:,:,:,0].cpu().data.numpy()
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link_batch = y_refiner[:,:,:,0].cpu().data.numpy()
    else:
        score_link_batch = y[:,:,:,1].cpu().data.numpy()

    boxes_list, polys_list, ret_scores = [], [], []
    for score_text, score_link in zip(score_text_batch, score_link_batch):
        # make score and link map
        # score_text = out[:, :, 0].cpu().data.numpy()
        # score_link = out[:, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

        # Xử lý kết quả
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)
        
        boxes_list.append(boxes)
        polys_list.append(polys)
        ret_scores.append(ret_score_text)

    return boxes_list, polys_list, ret_scores

def get_detector(pretrained, device='cpu', quantize=True, cudnn_benchmark=False):
    net = CRAFT()
    if device == 'cpu':
        net.load_state_dict(copyStateDict(torch.load(pretrained, map_location=device, weights_only=False)))
        if quantize:
            try:
                torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        net.load_state_dict(copyStateDict(torch.load(pretrained, map_location=device, weights_only=False)))
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = cudnn_benchmark
    net.eval()
    return net