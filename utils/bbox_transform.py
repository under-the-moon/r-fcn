import numpy as np


def bbox_transform(ex_rois, gt_rois):
    """
      返回生成的anchor和真实box之间的线性映射
      Gˆx =Pwdx(P)+Px (1)
      Gˆy =Phdy(P)+Py (2)
      Gˆw = Pw exp(dw(P))(3)
      Gˆh = Ph exp(dh(P))
      tx = (Gx − Px)/Pw (6)
      ty=(Gy−Py)/Ph (7)
      tw = log(Gw/Pw) (8)
      th = log(Gh/Ph).
      :param ex_rois:
      :param gt_rois:
      :return:
    """
    # 首先转换成中心点坐标
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1)

    dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    dw = np.log(gt_widths / ex_widths)
    dh = np.log(gt_heights / ex_heights)
    targets = np.vstack((dx, dy, dw, dh)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas):
    '''
    Applies deltas to box coordinates to obtain new boxes, as described by
    deltas
    '''
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[0] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[1] - 1), 0)
    # x2 < im_shape[0]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[0] - 1), 0)
    # y2 < im_shape[1]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[1] - 1), 0)
    return boxes
