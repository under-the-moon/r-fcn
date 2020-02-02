from exception import ValueValidException
from rpn.anchors01 import get_anchors as get_anchors01
from rpn.anchors02 import get_anchors as get_anchors02


class Anchors(object):

    def __init__(self, feature_size=(38, 50), feat_stride=16):
        """

        :param feature_size: 特征图大小 (height, width)
        :param feat_stride:
        """
        self.feature_size = feature_size
        self._feat_stride = feat_stride

    def get_anchors(self, mode='01'):
        """
        :param mode: 两种实现方式01和faster-rcnn 源码实现方式一致  02是另一种方式 结果和01是一致的
        :return:
        """
        if mode == '01':
            anchors = get_anchors01(self.feature_size, self._feat_stride)
        elif mode == '02':
            anchors = get_anchors02(self.feature_size, self._feat_stride)
        else:
            raise ValueValidException('Anchors model %s is not support' % mode)
        return anchors
