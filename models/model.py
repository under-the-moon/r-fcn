from config import Config
from exception import ValueValidException
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from models.r_roi_pooling import RRoiPooling

config = Config()

nets = config.network
im_dims = config.im_size
# 不包含背景
classes_num = config.classes_num
k = config.k


def get_model_body(input_tensor, net='vgg16', trainable=True):
    if net not in nets:
        raise ValueValidException('net: {%s} not supported, only support %s' % (net, nets))
    if net == config.network[0]:
        share_model = VGG16(input_tensor=input_tensor, include_top=False)
    elif net == config.network[1]:
        share_model = ResNet50(input_tensor=input_tensor, include_top=False)
    elif net == config.network[2]:
        share_model = InceptionV3(input_tensor=input_tensor, include_top=False)
    else:
        print('resnet101 still not implement')
    if not trainable:
        for layer in share_model.layers:
            layer.trainable = False
    model = Model(inputs=share_model.input, outputs=share_model.get_layer(index=-2).output)
    return model


def get_rpn_model(model_body, anchors_num):
    share_features = model_body.output
    output = Conv2D(256, 3, padding='same', activation='relu',
                    kernel_initializer='normal', name='merge_conv2d')(share_features)
    rpn_cls_output = Conv2D(anchors_num * 2, 1, kernel_initializer='uniform',
                            name='rpn_cls_conv2d')(output)
    rpn_reg_output = Conv2D(anchors_num * 4, 1, kernel_initializer='zero',
                            name='rpn_reg_conv2d')(output)
    rpn_model = Model(model_body.input, [rpn_cls_output, rpn_reg_output])
    return rpn_model


def get_rfcn_model(model_body, rois):
    share_features = model_body.output
    output = Conv2D(1024, 1, kernel_initializer='uniform', kernel_regularizer='l2',
                    name='reduce_channle_conv2d')(share_features)
    # get score map
    rfcn_cls_output = Conv2D(k * k * (classes_num + 1), 1, kernel_initializer='uniform', kernel_regularizer='l2',
                             name='rfcn_cls_output')(output)
    # get reg map
    # 这里如果使用和faster-rcnn 中最后一个维度一致的化 k * k * 4 * 9 则不需要修改数据集的构建
    # 这里因为只需要输出4 而不是faster-rcnn的4 * 9则需要都训练数据标签target_bbox 进行修改
    rfcn_reg_output = Conv2D(k * k * 4, 1, kernel_initializer='uniform', kernel_regularizer='l2',
                             name='rfcn_reg_output')(output)
    # roi
    rfcn_cls_output = RRoiPooling(im_dims=im_dims, last_dim=classes_num + 1, k=3)([rfcn_cls_output, rois])
    rfcn_reg_output = RRoiPooling(im_dims=im_dims, last_dim=4, k=3)([rfcn_reg_output, rois])
    rfcn_model = Model([model_body.input, rois], [rfcn_cls_output, rfcn_reg_output])
    return rfcn_model


# from tensorflow.keras.layers import Input
# import tensorflow as tf

# input = Input(shape=(600, 1000, 3))
# share_model = get_model_body(input)
# # share_model.summary()
#
# # rpn_model = get_rpn_model(share_model, 9)
# # rpn_model.summary()
# rois = Input(shape=(10, 5))
# # rois = tf.ones((1, 10, 5))
# rfcn_model = get_rfcn_model(share_model, rois)
# rfcn_model.summary()
