import tensorflow as tf
from tensorflow.keras.layers import Layer

"""
该层是用来对r-fcn进行roipooling
和faster-rcnn roipooling实现不同
"""


class RRoiPooling(Layer):
    def __init__(self, im_dims, last_dim, k=3, **kwargs):
        # k 是超参数 对 position-sensitive score maps or position-sensitive regression
        # 进行roi后feature maps大小
        super(RRoiPooling, self).__init__(**kwargs)
        self._im_dims = im_dims
        self._k = k
        # 因为cls 和 reg 都是用的这一层实现 所以这里将最后一维度穿进来
        # cls: C + 1  reg : 4
        self._last_dim = last_dim

    def call(self, inputs, **kwargs):
        """

        :param inputs: inputs shape (None, H, W, K * K * (C + 1)) or (None, H, W, K * K * 4)
        第一个是 position-sensitive score maps 第二个是 position-sensitive regression maps
        :param rois: (N, 5) (batch_id, x1, y1, x2, y2)
        :param kwargs:
        :return:
        """
        feature_maps, rois = inputs[0], inputs[1]
        # reshape 成rank: 2
        rois = tf.reshape(rois, (-1, 5))
        # 进行rois标准化
        k = self._k
        feature_split = tf.split(feature_maps, num_or_size_splits=k * k, axis=-1)
        # y1, x1, y2, x2
        boxes, batch_ids = self._normalize_boxes(rois)
        # 将boxes分成k * k个bin 每个bin大小为bin_w * bin_h
        bin_w = (boxes[..., 3] - boxes[..., 1]) / k
        bin_h = boxes[..., 2] - boxes[..., 0] / k
        sensitive_boxes = []
        for ih in range(k):
            for iw in range(k):
                box_coordinates = [boxes[..., 0] + ih * bin_h,
                                   boxes[..., 1] + iw * bin_w,
                                   boxes[..., 0] + (ih + 1) * bin_h,
                                   boxes[..., 1] + (iw + 1) * bin_h]
                sensitive_boxes.append(tf.stack(box_coordinates, axis=-1))
        features = []
        for (feature, box) in zip(feature_split, sensitive_boxes):
            # crop对于区域后resize 成 (2k, 2k) 这个大小可以调整 不要太离谱就行
            pooled_features = tf.image.crop_and_resize(feature, box, tf.cast(batch_ids, dtype=tf.int32), [k * 2, k * 2])
            features.append(pooled_features)
        # [(N, 6, 6, self._last_dim), (N, 6, 6, self._last_dim), (N, 6, 6, self._last_dim), ...] 总共9个
        # N 为 len(rois) == len(boxes)
        # 然后对list中的tesor各个位置上的值相加并求平均
        sensitive_features = tf.add_n(features) / len(features)
        # (N, self._last_dim)
        # 这里可以理解为进行投票 所有值想加得到类别概率
        output = tf.reduce_mean(sensitive_features, axis=[1, 2])
        output = tf.expand_dims(output, axis=0)
        return output

    def _normalize_boxes(self, rois):
        """
        对rois 进行normalize 使其满足tf.image.crop_and_resize方法参数格式
        :param rois: (None, 5)  batch_id, x1, y1, x2, y2
        :return:
        """
        im_dims = self._im_dims
        normalization = tf.cast(tf.stack([im_dims[1], im_dims[0], im_dims[1], im_dims[0]], axis=0),
                                dtype=tf.float32)
        batch_ids = rois[..., 0]
        boxes = rois[..., 1:]
        boxes = tf.div(boxes, normalization)
        # tf.stop_gradient
        # y1, x1, y2, x2
        boxes = tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)
        return boxes, batch_ids
