import imgproc
import torch
import cv2
import craft_utils
import numpy as np

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

    boxes_list, polys_list, ret_scores = [], [], []
    for out in y:
        # make score and link map
        score_text = out[:, :, 0].cpu().data.numpy()
        score_link = out[:, :, 1].cpu().data.numpy()

        if refine_net is not None:
            with torch.no_grad():
                out_refiner = refine_net(out, feature)
            score_link = out_refiner[0,:,:,0].cpu().data.numpy()

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