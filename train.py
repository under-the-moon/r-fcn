import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from voc_data import VocData
import os
from tensorflow.keras.callbacks import TensorBoard
from keras.utils.generic_utils import Progbar
import numpy as np
import time
from rpn.proposal_target_layer import proposal_target_layer
from rpn.proposal_layer import proposal_layer
from config import Config
from loss_func import rpn_cls_loss, rpn_reg_loss, rfcn_cls_loss, rfcn_reg_loss
from models.model import get_model_body, get_rpn_model, get_rfcn_model
from tensorflow.keras.layers import Input
# keras one hot
# tf.one_hot
from tensorflow.keras.utils import to_categorical

cfg = Config()
# load config
img_widht, img_height = cfg.im_size
anchors_num = cfg.anchors_num
classes_num = cfg.classes_num
keep_prob = cfg.keep_prob
pooled_height = cfg.pooled_height
pooled_width = cfg.pooled_width
im_size = cfg.im_size
rois_num = cfg.train_rpn_post_nms_top_n
batch_size = cfg.batch_size

# load data
voc_train_data = VocData('~/segment_data', 2007, 'train', './data/voc_classes.txt')
voc_train_g = voc_train_data.data_generator_wrapper(batch_size)
# voc_val_data = VocData('~/segment_data', 2007, 'val', './data/voc_classes.txt')
# voc_val_g = voc_val_data.data_generator_wrapper(batch_size)

# create model
input_tensor = Input(shape=(img_height, img_widht, 3))
input_rois = Input(shape=(None, 5))

model_body = get_model_body(input_tensor, net='vgg16', trainable=True)
# rpn model
rpn_model = get_rpn_model(model_body, anchors_num)
# rfcn model
rfcn_model = get_rfcn_model(model_body, input_rois)

log_path = './logs'
if not os.path.exists(log_path):
    os.mkdir(log_path)
rpn_callback = TensorBoard(os.path.join(log_path, '000'))
rpn_callback.set_model(rpn_model)

rfcn_callback = TensorBoard(os.path.join(log_path, '000'))
rfcn_callback.set_model(rfcn_model)

rpn_model_path = os.path.join(log_path, 'rpn_weights.h5')
rfcn_model_path = os.path.join(log_path, 'rfcn_model_weights.h5')

if os.path.exists(rpn_model_path):
    rpn_model.load_weights(rpn_model_path)
if os.path.exists(rfcn_model_path):
    rfcn_model.load_weights(rfcn_model_path)

rpn_optimizer = Adam(lr=1e-5)
rfcn_optimizer = Adam(lr=1e-5)
rpn_model.compile(optimizer=rpn_optimizer, loss=[rpn_cls_loss, rpn_reg_loss])
rfcn_model.compile(optimizer=rfcn_optimizer, loss=[rfcn_cls_loss, rfcn_reg_loss])


# tensorboard
def write_log(callback, names, logs, global_step):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        callback.writer.add_summary(summary, global_step)
        callback.writer.flush()


def save_weights(rpn_model, rfcn_model):
    if os.path.exists(rpn_model_path):
        os.remove(rpn_model_path)
    if os.path.exists(rfcn_model_path):
        os.remove(rfcn_model_path)
    rpn_model.save_weights(rpn_model_path)
    rfcn_model.save_weights(rfcn_model_path)


# 训练参数
epoch_length = voc_train_data.sample_nums
num_epochs = 10
train_step = 0
losses = np.zeros((epoch_length, 4))
best_loss = 100000

for epoch_num in range(num_epochs):
    iter_num = 0
    # keras progress bar
    progbar = Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
    while True:
        start_time = time.time()
        X, Y, gt_boxes = next(voc_train_g)
        # 去掉gt_boxes第一个维度
        gt_boxes = np.squeeze(gt_boxes, axis=0)
        # train rpn
        rpn_loss = rpn_model.train_on_batch(X, Y)
        # write_log(rpn_callback, ['Elapsed time', 'rpn_cls_loss', 'rpn_reg_loss'],
        #           [time.time() - start_time, rpn_loss[0], rpn_loss[1]], train_step)

        # train rfcn
        rpn_bbox_cls, rpn_bbox_pred = rpn_model.predict_on_batch(X)
        # get shared feature_maps
        # get fastrcnn_model 的训练数据集
        rois = proposal_layer(rpn_bbox_cls, rpn_bbox_pred, cfg.im_size, cfg.feat_stride, eval_mode=False)
        train_rois, labels, bbox_targets = proposal_target_layer(rois, gt_boxes, voc_train_data.classes_num)
        # 添加batch_size 维度
        train_rois = train_rois.reshape((1,) + train_rois.shape)
        rfcn_loss = rfcn_model.train_on_batch([X, train_rois], [labels, bbox_targets])
        # write_log(rfcn_callback, ['Elapsed time', 'rfcn_cls_loss', 'rfcn_reg_loss'],
        #           [time.time() - start_time, rfcn_loss[0], rfcn_loss[1]], train_step)

        # loss有三个值取前两个
        losses[iter_num, [0, 1]] = rpn_loss[0:2]
        losses[iter_num, [2, 3]] = rfcn_loss[0:2]
        train_step += 1
        iter_num += 1
        progbar.update(iter_num, [('rpn_cls_loss', rpn_loss[0]), ('rpn_reg_loss', rpn_loss[1]),
                                  ('rpn_cls_loss_mean', np.mean(losses[:iter_num, 0])),
                                  ('rpn_reg_loss_mean', np.mean(losses[:iter_num, 1])),
                                  ('rpn_total_loss_mean', np.mean(np.sum(losses[:iter_num, [0, 1]], axis=-1), axis=0)),
                                  ('rfcn_cls_loss', rfcn_loss[0]), ('rfcn_cls_loss', rfcn_loss[1]),
                                  ('rfcn_cls_loss_mean', np.mean(losses[:iter_num, 2])),
                                  ('rfcn_reg_loss_mean', np.mean(losses[:iter_num, 3])),
                                  ('rfcn_total_loss_mean', np.mean(np.sum(losses[:iter_num, [2, 3]], axis=-1), axis=0))
                                  ])

        # 采用最后一位指标来进行判断
        curr_loss = rpn_loss[2] + rfcn_loss[2]
        if iter_num % 100 == 0:
            if curr_loss < best_loss:
                best_loss = curr_loss
                save_weights(rpn_model, rfcn_model)

        if iter_num == epoch_length:
            if curr_loss < best_loss:
                best_loss = curr_loss
                save_weights(rpn_model, rfcn_model)
            # 推出while 循环开始下一个epoch
            break
