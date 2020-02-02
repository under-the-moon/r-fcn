import numpy as np
from rpn.generate_anchors import generate_anchors

"""
得到所有的anchors
总共实现了两种方式
这是一种实现方式  采用和faster-rcnn源码一样的实现方式
另一种方式见 anchors02.py
"""


def get_anchors(feature_maps_shape, feat_stride=16):
    anchors = generate_anchors()
    num_anchors = anchors.shape[0]
    feature_height, feature_width = feature_maps_shape
    shift_x = np.arange(0, feature_width) * feat_stride
    shift_y = np.arange(0, feature_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # A : 9
    A = num_anchors
    # H * W
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    return all_anchors, A
