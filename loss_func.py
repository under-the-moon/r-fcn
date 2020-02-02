import tensorflow as tf
from config import Config

config = Config()


def rpn_cls_loss(y_true, y_pred):
    rpn_cls_output = y_pred
    rpn_labels = y_true
    # 计算分类损失
    # (1, H, W, 2K) --> (1, 2K, H, W) --> (1, 2, K * H, W ) --> (1, K * H, W, 2 )
    # 这里是和label进行了一样的变换 如果不做这些变换 会变成 1张背景 1张前景
    shape = rpn_cls_output.shape
    rpn_cls_output = tf.transpose(rpn_cls_output, [0, 3, 1, 2])
    rpn_cls_output = tf.reshape(rpn_cls_output, [-1, 2, shape[3] // 2 * shape[1], shape[2]])
    rpn_cls_output = tf.transpose(rpn_cls_output, [0, 2, 3, 1])
    rpn_cls_output = tf.reshape(rpn_cls_output, [-1, 2])

    rpn_labels = tf.reshape(rpn_labels, [-1])
    rpn_labels = tf.cast(rpn_labels, dtype=tf.int32)
    rpn_labels = tf.reshape(tf.gather(rpn_labels, tf.where(tf.not_equal(rpn_labels, -1))), [-1])
    rpn_cls_output = tf.reshape(tf.gather(rpn_cls_output, tf.where(tf.not_equal(rpn_labels, -1))), [-1, 2])
    rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_output,
                                                                                      labels=rpn_labels))
    return rpn_cross_entropy


def rpn_reg_loss(y_true, y_pred):
    # 计算边框回归损失
    rpn_reg_output = y_pred
    # (1, A * 4, H, W)
    rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.split(y_true, num_or_size_splits=3, axis=1)
    rpn_bbox_targets = tf.transpose(rpn_bbox_targets, (0, 2, 3, 1))
    rpn_bbox_inside_weights = tf.transpose(rpn_bbox_inside_weights, (0, 2, 3, 1))
    rpn_bbox_outside_weights = tf.transpose(rpn_bbox_outside_weights, (0, 2, 3, 1))
    diff = tf.multiply(rpn_bbox_inside_weights, rpn_reg_output - rpn_bbox_targets)
    diff_sl1 = smooth_l1(diff, 3.0)
    rpn_bbox_reg = tf.reduce_sum(tf.multiply(rpn_bbox_outside_weights, diff_sl1))
    rpn_bbox_reg_losses = config.train_rpn_bbox_lambda * rpn_bbox_reg
    return rpn_bbox_reg_losses


def rfcn_cls_loss(y_true, y_pred):
    logits = y_pred
    labels = y_true
    logits = tf.reshape(logits, [-1, logits.shape[-1]])
    labels = tf.reshape(labels, [-1])
    labels = tf.cast(labels, tf.int32)
    rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                      labels=labels))
    return rpn_cross_entropy


def rfcn_reg_loss(y_true, y_pred):
    fast_rcnn_bbox_pred = y_pred
    bbox_targets, roi_inside_weights, roi_outside_weights = tf.split(y_true, num_or_size_splits=3, axis=-1)
    diff = tf.multiply(roi_inside_weights, fast_rcnn_bbox_pred - bbox_targets)
    diff_sL1 = smooth_l1(diff, 1.0)
    # Only count loss for positive anchors
    roi_bbox_reg = tf.reduce_mean(tf.reduce_sum(tf.multiply(roi_outside_weights, diff_sL1), axis=1))

    # Constant for weighting bounding box loss with classification loss
    roi_bbox_reg = config.train_rfcn_bbox_lambda * roi_bbox_reg

    return roi_bbox_reg


def smooth_l1(x, sigma):
    '''
                      0.5 * (sigma * x)^2  if |x| < 1/sigma^2
      smoothL1(x) = {
                      |x| - 0.5/sigma^2    otherwise
    '''

    conditional = tf.less(tf.abs(x), 1 / sigma ** 2)
    close = 0.5 * (sigma * x) ** 2
    far = tf.abs(x) - 0.5 / sigma ** 2
    return tf.where(conditional, close, far)
