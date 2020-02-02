"""
源码使用cpython实现 这里我就不用cpython了 pycharm社区版写cpython没有代码提示
这里就用Python实现 对cpython怎么实现有兴趣的可以去参考源码
"""
import numpy as np


def bbox_overlaps(boxes, query_boxes):
    """

    :param boxes: shape(None, 4)
    :param query_boxes: (None, 4)
    :return:
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(N):
            # 计算query_box面积
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    # iw * ih 是相交部分的面积
                    # ua: area(A) + area(B) - area(AB)
                    ua = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih
                    overlaps[n, k] = iw * ih / ua
    return overlaps
