"""
proposal_layer 计算出参与roi_pooling的regions
"""
import numpy as np
from rpn.anchors import Anchors
from config import Config
from utils.bbox_transform import bbox_transform_inv, clip_boxes
from utils.nms import py_cpu_nms as nms
import tensorflow as tf
from utils.softmax import rpn_softmax

cfg = Config()


def proposal_layer_(args, im_size, feat_stride=16, eval_mode=False):
    return tf.py_function(_proposal_layer, [*args, im_size, feat_stride, eval_mode], [tf.float32])


def proposal_layer(rpn_bbox_cls, rpn_bbox_pred, im_size, feat_stride=16, eval_mode=False):
    return _proposal_layer(rpn_bbox_cls, rpn_bbox_pred, im_size, feat_stride, eval_mode)


def _proposal_layer(rpn_bbox_cls, rpn_bbox_pred, im_size, feat_stride, eval_mode):
    """

    :param rpn_bbox_cls: (None, H, W, 2 * k)
    :param rpn_bbox_pred: (None, H, W, 4 * k)
    :param im_size: (800, 600)
    :param feat_stride: 16
    :return:
    """
    rpn_bbox_cls_prob = rpn_softmax(rpn_bbox_cls)
    anchor = Anchors(feat_stride=feat_stride)
    # all_anchors (A * H * W, 4)
    anchors, A = anchor.get_anchors()
    num_anchors = A
    rpn_bbox_cls_prob = np.transpose(rpn_bbox_cls_prob, [0, 3, 1, 2])
    rpn_bbox_pred = np.transpose(rpn_bbox_pred, [0, 3, 1, 2])

    assert rpn_bbox_cls_prob.shape[0] == 1, 'Only support 1 batch_size'

    if not eval_mode:
        # 训练模式
        pre_nms_topN = cfg.train_rpn_pre_nms_top_n
        post_nms_topN = cfg.train_rpn_post_nms_top_n
        nms_thresh = cfg.train_rpn_nms_thresh
        min_size = cfg.train_rpn_min_size
    else:
        # 验证模式
        pre_nms_topN = cfg.test_rpn_pre_nms_top_n
        post_nms_topN = cfg.test_rpn_post_nms_top_n
        nms_thresh = cfg.test_rpn_nms_thresh
        min_size = cfg.test_rpn_min_size
    scores = rpn_bbox_cls_prob[:, num_anchors:, :, :]
    bbox_deltas = rpn_bbox_pred
    # (1, 4 * k, H, W) -> (1, H, W, 4 * A)
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
    # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, im_size)

    # 3. remove predicted boxes with either height or width < threshold
    keep = _filter_boxes(proposals, min_size)
    proposals = proposals[keep, :]
    scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    keep = nms(np.hstack((proposals, scores)), nms_thresh)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    # scores = scores[keep]

    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    return blob


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
