"""
为 fast-rcnn生成训练数据
"""
import numpy as np
from config import Config
from utils.bbox_overlaps import bbox_overlaps
from utils.bbox_transform import bbox_transform

config = Config()


def proposal_target_layer(rpn_rois, gt_boxes, classes_num):
    rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
        _proposal_target_layer_py(rpn_rois, gt_boxes, classes_num)
    bbox_targets = np.concatenate([bbox_targets, bbox_inside_weights, bbox_outside_weights], axis=-1)
    labels = np.expand_dims(labels, axis=0)
    bbox_targets = np.expand_dims(bbox_targets, axis=0)
    return rois, labels, bbox_targets


def _proposal_target_layer_py(rpn_rois, gt_boxes, classes_num):
    all_rois = rpn_rois
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    # Include ground-truth boxes in the set of candidate rois
    # 将ground_th加入候选区域用于训练
    all_rois = np.vstack(
        (all_rois, np.hstack((zeros, gt_boxes[:, :4])))
    )
    # Sanity check: single batch only
    assert np.all(all_rois[:, 0] == 0), \
        'Only single item batches are supported'
    # 为每张图片设置正负样本数目
    num_images = 1
    rois_per_image = config.train_batch_size // num_images
    # 128 * 0.25 = 32
    fg_rois_per_image = np.round(config.train_fg_fraction * rois_per_image).astype(np.int32)

    # Sample rois with classification labels and bounding box regression
    # targets
    # 生成训练用的labels 和 边框回归数据
    labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, gt_boxes, fg_rois_per_image,
        rois_per_image, classes_num)

    rois = rois.reshape(-1, 5)
    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, 4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, 4)
    # 正负样本权重
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
    return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """
    Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # 这里是将数组装进连续内存并计算iou
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    # 为每个anchor设置所属类别  与哪个gt_boxes相交iou最大就是对应的class
    labels = gt_boxes[gt_assignment, 4]

    # 这里是设置正负样本数目
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= config.train_fg_thresh)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        # 随机抽样
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < config.train_bg_thresh_hi) &
                       (max_overlaps >= config.train_bg_thresh_lo))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
    # The indices that we're selecting (both fg and bg)
    # 得到
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    # labels的size 为 128
    labels = labels[keep_inds]

    # Clamp labels for the background RoIs to 0
    # 前32个是正样本  后面的都是负样本 0表示背景
    labels[fg_rois_per_this_image:] = 0
    # 128个
    rois = all_rois[keep_inds]
    # 将候选区域根据坐标回归公式进行转换
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
    # 生成坐标回归用的训练数据
    # 将 n * 5 -> n * 4k (k是class_num)
    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).
    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        bbox_targets[ind, :] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, :] = (1, 1, 1, 1)
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if config.train_bbox_normalize_targets_precomputed:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(config.train_bbox_normalize_means))
                   / np.array(config.train_bbox_normalize_stds))
    #     将类别拼接到第一维
    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)
