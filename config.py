class Config(object):

    def __init__(self):
        # 支持的network
        self.network = ['vgg16', 'resnet50', 'inceptionv3', 'resnet101']
        # 可以进行的角度旋转
        self.rotate_angles = [90, 180, 270]
        self.anchors_num = 9
        # 不包含背景的类别数
        self.classes_num = 20
        # 位置维度 roipooling后的特征大小
        self.k = 3
        # dropout
        self.keep_prob = .5
        # batch_size
        self.batch_size = 1

        # 图片大小
        # 为了方便就reisze成固定大小
        self.im_size = (800, 608)
        self.feat_stride = 16
        self.pooled_width, self.pooled_height = 7, 7
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0

        # anchor box scales
        self.anchor_box_scales = [128, 256, 512]
        # anchor box ratios
        self.anchor_box_ratios = [0.5, 1, 2]
        # If an anchor satisfied by positive and negative conditions set to negative
        self.train_rpn_clobber_positives = False
        # IOU < thresh: negative example
        self.train_rpn_negative_overlap = 0.3
        # IOU >= thresh: positive example
        self.train_rpn_positive_overlap = 0.7
        # Total number of examples
        self.train_rpn_batch_size = 256
        # Max number of foreground examples
        self.train_rpn_fg_fraction = 0.5
        # Give the positive RPN examples weight of p * 1 / {num positives}
        # and give negatives a weight of (1 - p)
        # Set to -1.0 to use uniform example weighting
        self.train_rpn_positive_weight = -1.0

        # NMS threshold used on RPN proposals
        self.train_rpn_nms_thresh = 0.7
        # Number of top scoring boxes to keep before apply NMS to RPN proposals
        self.train_rpn_pre_nms_top_n = 12000
        # Number of top scoring boxes to keep after applying NMS to RPN proposals
        self.train_rpn_post_nms_top_n = 2000
        # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
        self.train_rpn_min_size = 16

        # NMS threshold used on RPN proposals
        self.test_rpn_nms_thresh = 0.7
        # Number of top scoring boxes to keep before apply NMS to RPN proposals
        self.test_rpn_pre_nms_top_n = 6000
        # Number of top scoring boxes to keep after applying NMS to RPN proposals
        self.test_rpn_post_nms_top_n = 300
        # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
        self.test_rpn_min_size = 16

        # loss
        self.train_rpn_bbox_lambda = 10.0
        # Relative weight of Fast RCNN bounding box loss
        self.train_rfcn_bbox_lambda = 1.0

        # Minibatch size (number of regions of interest [ROIs])
        self.train_batch_size = 128
        # Fraction of minibatch that is labeled foreground (i.e. class > 0)
        self.train_fg_fraction = 0.25

        # Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
        self.train_fg_thresh = 0.5

        # Overlap threshold for a ROI to be considered background (class = 0 if
        # overlap in [LO, HI))
        self.train_bg_thresh_hi = 0.5
        self.train_bg_thresh_lo = 0.0

        # Normalize the targets using "precomputed" (or made up) means and stdevs
        # (BBOX_NORMALIZE_TARGETS must also be True)
        self.train_bbox_normalize_targets_precomputed = False
        self.train_bbox_normalize_means = (0.0, 0.0, 0.0, 0.0)
        self.train_bbox_normalize_stds = (0.1, 0.1, 0.2, 0.2)
