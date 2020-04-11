import cv2
import numpy as np
from config import Config
from exception import ValueValidException


class DataAugment(object):

    def __init__(self, augment=True, horizontal_flip=False, vertical_flip=False,
                 rotate_angle=False):
        """

        :param image_path: 图片地址
        :param gt_boxes: 真实anchors
        :param augment: 是否进行数据增广
        :param horizontal_flip: 是否水平旋转
        :param vertical_flip: 是否垂直旋转
        :prams rotate_angle 是否进行角度旋转
        """
        self._augment = augment
        self._horizontal_flip = horizontal_flip
        self._vertical_flip = vertical_flip
        self._rotate_angle = rotate_angle
        self._cfg = Config()

    def __call__(self, image_path, gt_boxes, **kwargs):
        img = cv2.imread(image_path)
        # height/width/channel
        height, width = img.shape[:2]
        if self._augment:
            flip = np.random.rand()
            if self._horizontal_flip and flip < 0.5:
                img = cv2.flip(img, 1)
                gt_boxes[:, [0, 2]] = width - gt_boxes[:, [2, 0]]
            # 水平翻转后就不垂直翻转
            if self._vertical_flip and flip > 0.5:
                img = cv2.flip(img, 0)
                gt_boxes[:, [1, 3]] = height - gt_boxes[:, [3, 1]]
            # 是否旋转90度
            rotate_angle_flip = np.random.rand()
            if self._rotate_angle and rotate_angle_flip < 0.5:
                angle = np.random.choice(self._cfg.rotate_angles, 1)[0]
                if angle == 270:
                    img = np.transpose(img, (1, 0, 2))
                    img = cv2.flip(img, 0)
                    gt_boxes[:, [0, 2]] = gt_boxes[:, [1, 3]]
                    gt_boxes[:, [1, 3]] = width - gt_boxes[:, [2, 0]]
                elif angle == 180:
                    img = cv2.flip(img, -1)
                    gt_boxes[:, [0, 2]] = width - gt_boxes[:, [2, 0]]
                    gt_boxes[:, [1, 3]] = width - gt_boxes[:, [3, 1]]
                elif angle == 90:
                    img = np.transpose(img, (1, 0, 2))
                    img = cv2.flip(img, 1)
                    gt_boxes[:, [1, 3]] = gt_boxes[:, [0, 2]]
                    gt_boxes[:, [0, 2]] = height - gt_boxes[:, [2, 0]]
                else:
                    raise ValueValidException('rotate %s angle is not defined' % angle)
        return img, gt_boxes
