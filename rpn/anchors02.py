"""
得到所有的anchors
总共实现了两种方式
这是一种实现方式
另一种方式见 anchors01.py
"""
import numpy as np
from config import Config

cfg = Config()


def get_anchors(feature_maps_shape, feat_stride=16):
    height, width = feature_maps_shape
    anchor_sizes = cfg.anchor_box_scales
    anchor_ratios = cfg.anchor_box_ratios
    anchors_num = len(anchor_sizes) * len(anchor_ratios)
    anchors = np.zeros(shape=(anchors_num, height, width, 4), dtype=np.float32)
    for size_idx in range(len(anchor_sizes)):
        for ratio_idx in range(len(anchor_ratios)):
            w, h = anchor_sizes[size_idx], anchor_sizes[size_idx]
            ratio = anchor_ratios[ratio_idx]
            size = w * h
            size_ratio = size / ratio
            anchor_w = np.round(np.sqrt(size_ratio))
            anchor_h = np.round(anchor_w * ratio)
            for iy in range(height):
                y1 = feat_stride * (iy + 0.5) - anchor_h / 2
                y2 = feat_stride * (iy + 0.5) + anchor_h / 2
                for ix in range(width):
                    x1 = feat_stride * (ix + 0.5) - anchor_w / 2
                    x2 = feat_stride * (ix + 0.5) + anchor_w / 2
                    anchors[size_idx * len(anchor_sizes) + ratio_idx, iy, ix, :] = [x1, y1, x2, y2]
    # (A, H, W, 4)
    K = height * width
    A = anchors_num
    anchors = anchors.reshape((A, K, 4)).transpose((1, 0, 2)).reshape((K * A, 4))
    return anchors, A
