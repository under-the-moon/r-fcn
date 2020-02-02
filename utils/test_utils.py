import cv2
import numpy as np
from utils.bbox_transform import bbox_transform_inv


def preprocess_test_img(img, config):
    # resize img
    height, width, _ = img.shape
    im_size = config.im_size
    ratios = (width / im_size[0], height / im_size[1])
    img = cv2.resize(img, im_size, interpolation=cv2.INTER_CUBIC)
    # bgr - rgb
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= config.img_channel_mean[0]
    img[:, :, 1] -= config.img_channel_mean[1]
    img[:, :, 2] -= config.img_channel_mean[2]
    img /= config.img_scaling_factor
    img = np.expand_dims(img, axis=0)
    return img, ratios


def bbox_reg_target(fastrcnn_reg_output, labels, rois, scores):
    """
    得到修正后的边框
    :param fastrcnn_reg_output:
    :param labels:
    :param rois:
    :return:
    """
    inds = np.where(labels[labels > 0])[0]
    bbox_reg = np.zeros((len(inds), 5))
    for ind in inds:
        cls = labels[ind]
        start = (int(cls) - 1) * 4
        end = start + 4
        bbox_reg[ind, 0] = cls
        bbox_reg[ind, 1:] = fastrcnn_reg_output[ind, start:end]
    rois = rois[inds, ...]
    # len(rois) == len(bbox_reg)
    pred_boxes = bbox_transform_inv(rois[:, 1], bbox_reg[:, 1:])
    # (None, 6) x1, y1, x2, y2, score, cls
    final_pred_boxes = np.hstack((pred_boxes, scores[inds, np.newaxis], labels[inds, np.newaxis]))
    return final_pred_boxes
