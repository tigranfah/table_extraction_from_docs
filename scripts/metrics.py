import tensorflow as tf
from tensorflow.keras import backend as K


def iou(pr, gt, eps=1e-6, threshold=None, activation='sigmoid'):

    intersection = tf.reduce_sum(gt * pr)
    union = tf.reduce_sum(gt) + tf.reduce_sum(pr) - intersection

    return intersection / union


def f1_score(pr, gt, beta=1, eps=1e-6):
    tp = tf.reduce_sum(gt * pr)
    fp = tf.reduce_sum(pr) - tp
    fn = tf.reduce_sum(gt) - tp

    return ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)


def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth