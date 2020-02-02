"""
Generates a regular grid of multi-scale, multi-aspect anchor boxes.
生成初始的9个anchor  生成方法不唯一  这里使用源码中的方法
"""
import numpy as np


def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2 ** np.arange(3, 6)):
    # 生成base anchor [0, 0, 15, 15]
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratios_anchors = _ratios_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratios_anchors[i], scales) for i in range(ratios_anchors.shape[0])])
    return anchors


def _ratios_enum(anchor, ratios):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    # 这里是求解方程得到的  x * y = w * h  x / y = ratios
    # x, y = np.sqrt(w * h / ratios), np.sqrt(w * h / ratios) * ratios
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _whctrs(anchor):
    """
    转换anchor坐标 (x1, y1, x2, y2) -> (w, h, x_center, y_center)
    :param anchor:
    :return:
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    import time

    t = time.time()
    a = generate_anchors()
    print(a)
    print(time.time() - t)
    # from IPython import embed; embed()
