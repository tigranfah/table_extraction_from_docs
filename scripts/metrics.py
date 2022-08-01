import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


def calculate_metrics(batch_tgt, batch_pred):
    iou_value = np.mean([iou(gt, pr) for pr, gt in zip(batch_pred, batch_tgt)])
    f1_score_value = np.mean([f1_score(gt, pr) for pr, gt in zip(batch_pred, batch_tgt)])
    precision_value = np.mean([precision(gt, pr) for pr, gt in zip(batch_pred, batch_tgt)])
    recall_value = np.mean([recall(gt, pr) for pr, gt in zip(batch_pred, batch_tgt)])

    return (
        iou_value,
        f1_score_value,
        precision_value,
        recall_value
    )


def recall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    TP_FN = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (TP_FN + K.epsilon())
    return recall


def precision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    TP_FP = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = TP / (TP_FP + K.epsilon())
    return precision


def iou(gt, pr, eps=1e-6):

    intersection = tf.reduce_sum(gt * pr)
    union = tf.reduce_sum(gt) + tf.reduce_sum(pr) - intersection

    return intersection / (union + eps)


def f1_score(gt, pr, beta=1, eps=1e-6):
    tp = tf.reduce_sum(gt * pr)
    fp = tf.reduce_sum(pr) - tp
    fn = tf.reduce_sum(gt) - tp

    return ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)


def jaccard_distance(y_true, y_pred, smooth=100):
    """ Calculates mean of Jaccard distance as a loss function """
    # print(y_true.shape, y_pred.shape)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd =  (1 - jac) * smooth
    return tf.reduce_mean(jd)


# a = tf.constant([[
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1]
# ]], dtype=tf.float32)

# b = tf.constant([[
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 0]
# ]], dtype=tf.float32)

# print(jaccard_distance(a, b))