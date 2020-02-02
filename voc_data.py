import numpy as np
from voc_annotation import VOCAnnotation
from utils.data_augment import DataAugment
from rpn.anchors import Anchors
from config import Config
from utils.bbox_overlaps import bbox_overlaps
from utils.bbox_transform import bbox_transform
import cv2
import os


class VocData(object):

    def __init__(self, data_path, year, mode, class_names_path, random_shuffle=True, data_augment=True):
        data_path = os.path.expanduser(data_path)
        voc_annotation = VOCAnnotation(data_path, year, mode, class_names_path)
        annotation_path = './data/%s_%s.txt' % (year, mode)
        # 没有背景
        self.class_names = voc_annotation.class_names
        self._annotations = self._parse_annotation_path(annotation_path)
        self.sample_nums = len(self._annotations)
        self._shuffle = random_shuffle
        if self._shuffle:
            self._random_shuffle()
        # 是否需要数据增广
        self._data_augment = data_augment
        self._cfg = Config()

    @property
    def classes_num(self):
        return len(self.class_names)

    def _random_shuffle(self):
        x = np.random.permutation(self.sample_nums)
        self._annotations = self._annotations[x]

    def _parse_annotation_path(self, annotation_path):
        annotations = open(annotation_path).readlines()
        annotations = [annotation.strip() for annotation in annotations]
        return np.array(annotations)

    def data_generator_wrapper(self, batch_size=1):
        assert batch_size == 1, 'batch_size should be one'
        return self._data_generator(batch_size)

    def _data_generator(self, batch_size):
        data_augment = DataAugment(augment=self._data_augment, horizontal_flip=True, vertical_flip=True)
        im_size = self._cfg.im_size
        feat_stride = self._cfg.feat_stride
        feature_width, feature_height = round(im_size[0] / feat_stride), round(im_size[1] / feat_stride)
        anchor = Anchors(feature_size=(feature_height, feature_width), feat_stride=feat_stride)
        i = 0
        while True:
            for annotation in self._annotations:
                img_data = []
                rpn_labels = []
                rpn_bbox_targets = []
                rpn_bbox_inside_weights = []
                rpn_bbox_outside_weights = []
                total_gt_boxes = []
                for b in range(batch_size):
                    if i == 0 and self._shuffle:
                        self._random_shuffle()
                    image_path, gt_boxes = self._parse_annotation(annotation)
                    # 进行数据增广防止过拟合
                    img, gt_boxes = data_augment(image_path, gt_boxes)
                    height, width = img.shape[:2]
                    # resize img
                    img = cv2.resize(img, im_size, interpolation=cv2.INTER_CUBIC)
                    # BGR -> RGB
                    img = img[:, :, (2, 1, 0)]
                    img = img.astype(np.float32)
                    img[:, :, 0] -= self._cfg.img_channel_mean[0]
                    img[:, :, 1] -= self._cfg.img_channel_mean[1]
                    img[:, :, 2] -= self._cfg.img_channel_mean[2]
                    img /= self._cfg.img_scaling_factor
                    img_data.append(img)
                    # reisze gt_boxes
                    gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * (im_size[0] / width)
                    gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * (im_size[1] / height)
                    # get anchors
                    all_anchors, A = anchor.get_anchors()
                    # 得到总额anchros数目 h * w * k (50 * 38 * 9)
                    total_anchors_num = len(all_anchors)
                    # 过滤 anchors
                    allow_border = 0
                    inds_inside = np.where((all_anchors[:, 0] >= allow_border) &
                                           (all_anchors[:, 1] >= allow_border) &
                                           (all_anchors[:, 2] <= (im_size[1] + allow_border)) &
                                           (all_anchors[:, 3] <= (im_size[0] + allow_border)))[0]
                    anchors = all_anchors[inds_inside, :]
                    labels = np.empty((len(inds_inside, )), dtype=np.float32)
                    labels.fill(-1)
                    # 计算iou
                    overlaps = bbox_overlaps(np.ascontiguousarray(anchors), np.ascontiguousarray(gt_boxes))
                    argmax_overlaps = np.argmax(overlaps, axis=1)
                    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
                    gt_argmax_overlaps = np.argmax(overlaps, axis=0)
                    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
                    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
                    if not self._cfg.train_rpn_clobber_positives:
                        labels[max_overlaps < self._cfg.train_rpn_negative_overlap] = 0
                    labels[gt_argmax_overlaps] = 1
                    labels[max_overlaps > self._cfg.train_rpn_positive_overlap] = 1
                    if self._cfg.train_rpn_clobber_positives:
                        labels[max_overlaps < self._cfg.train_rpn_negative_overlap] = 0
                    # 防止每张图片训练数据过多  这里每张图片positive + negative 的样本数等于256
                    # 其中尽量保证 positive和negative样本数目一致 如果正样本不够128则负样本增加满足一种图片取256个样本
                    num_fg = int(self._cfg.train_rpn_fg_fraction * self._cfg.train_rpn_batch_size)
                    fg_inds = np.where(labels == 1)[0]
                    if len(fg_inds) > num_fg:
                        # 这个表示随机采样 replace=false表示没有重复采样
                        disabled_inds = np.random.choice(fg_inds, size=len(fg_inds) - num_fg, replace=False)
                        labels[disabled_inds] = -1
                    num_bg = self._cfg.train_rpn_batch_size - np.sum(labels == 1)
                    bg_inds = np.where(labels == 0)[0]
                    if len(bg_inds) > num_bg:
                        disabled_inds = np.random.choice(bg_inds, size=len(bg_inds) - num_bg, replace=False)
                        labels[disabled_inds] = -1

                    bbox_targets = self._compute_targets(anchors, gt_boxes[argmax_overlaps, :])
                    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
                    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
                    bbox_inside_weights[labels == 1, :] = np.array([1.0, 1.0, 1.0, 1.0])
                    if self._cfg.train_rpn_positive_weight < 0:
                        # uniform weighting of examples (given non-uniform sampling) 非均匀采样
                        # 这里相当于把样本进行均匀采样处理出现的概率都是一样的权重是一样
                        num_examples = np.sum(labels >= 0)
                        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
                        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
                    else:
                        assert ((self._cfg.train_rpn_positive_weight > 0) &
                                (self._cfg.train_rpn_positive_weight < 1))
                        # 如果是非均匀采样这里将权重设置成 该样本的概率乘以1/positive_samples  loss = p * loss(positive) + (1-p) loss(negative)
                        # 目的就是调节正负样本损失在总损失中站的比例 防止样本多的损失占比过大
                        positive_weights = (self._cfg.train_rpn_positive_weight / np.sum(labels == 1))
                        negative_weights = ((1.0 - self._cfg.train_rpn_positive_weight) / np.sum(labels == 0))
                    # 给训练样本进行权重赋值
                    bbox_outside_weights[labels == 1] = positive_weights
                    bbox_outside_weights[labels == 0] = negative_weights
                    labels = self._unmap(labels, total_anchors_num, inds_inside, fill=-1)
                    # 把图像内部的anchor对应的bbox_target映射回所有的anchor(加上了那些超出边界的anchor，填充0)
                    bbox_targets = self._unmap(bbox_targets, total_anchors_num, inds_inside, fill=0)
                    # 把图像内部的anchor对应的bbox_target映射回所有的anchor(加上了那些超出边界的anchor，填充0)
                    # [H * W * A, 4]
                    bbox_inside_weights = self._unmap(bbox_inside_weights, total_anchors_num, inds_inside, fill=0)
                    # 把图像内部的anchor对应的bbox_target映射回所有的anchor(加上了那些超出边界的anchor，填充0)
                    bbox_outside_weights = self._unmap(bbox_outside_weights, total_anchors_num, inds_inside, fill=0)

                    # 进行reshape
                    #  [H * W * A] --> [H, W, A] --> [A, H, W]
                    labels = labels.reshape((feature_height, feature_width, A)).transpose((2, 0, 1))
                    labels = labels.reshape((A * feature_height, feature_width))
                    #  (H * W * A, 4) -> (H, W, A * 4) -> (A * 4, H, W)
                    bbox_targets = bbox_targets.reshape((feature_height, feature_width, A * 4)).transpose((2, 0, 1))
                    #  (H * W * A, 4) -> (H, W, A * 4) -> (A * 4, H, W)
                    bbox_inside_weights = bbox_inside_weights.reshape((feature_height, feature_width, A * 4)).\
                        transpose((2, 0, 1))
                    #  (H * W * A, 4) -> (H, W, A * 4) -> (A * 4, H, W)
                    bbox_outside_weights = bbox_outside_weights.reshape((feature_height, feature_width, A * 4)).\
                        transpose((2, 0, 1))
                    rpn_labels.append(labels)
                    rpn_bbox_inside_weights.append(bbox_inside_weights)
                    rpn_bbox_outside_weights.append(bbox_outside_weights)
                    rpn_bbox_targets.append(bbox_targets)
                    total_gt_boxes.append(gt_boxes)
                    i = (i + 1) % self.sample_nums
                rpn_labels = np.array(rpn_labels)
                rpn_bbox_inside_weights = np.array(rpn_bbox_inside_weights)
                rpn_bbox_targets = np.array(rpn_bbox_targets)
                rpn_bbox_outside_weights = np.array(rpn_bbox_outside_weights)
                img_data = np.array(img_data)
                total_gt_boxes = np.asarray(total_gt_boxes)
                # (1, 108, 38, 50)
                rpn_bbox_targets = np.concatenate([rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights],
                                                  axis=1)
                yield img_data, [rpn_labels, rpn_bbox_targets], total_gt_boxes

    def _parse_annotation(self, annotation):
        lines = annotation.strip().split()
        image_path = lines[0]
        gt_boxes = [list(map(float, box.split(','))) for box in lines[1:]]
        return image_path, np.asarray(gt_boxes)

    def _compute_targets(self, ex_rois, gt_rois):
        assert ex_rois.shape[0] == gt_rois.shape[0]
        assert ex_rois.shape[1] == 4
        assert gt_rois.shape[1] == 5
        return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

    def _unmap(self, data, count, inds, fill=0):
        if len(data.shape) == 1:
            ret = np.empty(count, dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret
#
# voc_data = VocData('~/segment_data', 2007, 'train', './data/voc_classes.txt')
# g = voc_data.data_generator_wrapper()
# img_data, [rpn_labels, rpn_bbox_targets], total_gt_boxes = next(g)
#
# print(img_data.shape)
# print(rpn_labels)
# print(rpn_labels.shape)
# print(rpn_bbox_targets.shape)
# print(total_gt_boxes.shape)
