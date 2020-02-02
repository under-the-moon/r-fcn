import numpy as np


def rpn_softmax(rpn_cls_ouput):
    """
    (1, H, W, 2K) --> (1, 2K, H, W) --> (1, 2, K * H, W ) --> (1, K * H, W, 2 )
    (1, K * H, W, 2) --> (1, 2, K * H, W ) --> (1, 2K, H, W) --> (1, H, W, 2K)
    :param args: (n,H,W,18)
    :return:
    """
    shape = rpn_cls_ouput.shape
    rpn_cls_ouput = np.transpose(rpn_cls_ouput, [0, 3, 1, 2])
    rpn_cls_ouput = np.reshape(rpn_cls_ouput, [-1, 2, shape[3] // 2 * shape[1], shape[2]])
    rpn_cls_ouput = np.transpose(rpn_cls_ouput, [0, 2, 3, 1])
    rpn_cls_prob = softmax(rpn_cls_ouput)
    # Reshape back to the original
    rpn_cls_prob = np.transpose(rpn_cls_prob, [0, 3, 1, 2])
    rpn_cls_prob = np.reshape(rpn_cls_prob, [shape[0], shape[3], shape[1], shape[2]])
    rpn_cls_prob = np.transpose(rpn_cls_prob, [0, 2, 3, 1])
    return rpn_cls_prob


def softmax(cls_output):
    rpn_cls_output = np.exp(cls_output)
    rpn_cls_output = rpn_cls_output / np.sum(rpn_cls_output, axis=-1, keepdims=True)
    return rpn_cls_output
