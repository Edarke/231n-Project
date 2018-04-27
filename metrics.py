import tensorflow as tf


def soft_dice(predictions, labels):
    # 2TP / (T - TP + 2TP + P - TP)
    predictions = tf.reshape(predictions, [-1])
    truth = tf.reshape(labels, [-1])
    intersection = 2*tf.reduce_sum(predictions * truth)
    union = tf.reduce_sum(predictions + truth)
    union = tf.Print(union, [intersection, union])
    return 1 - intersection/(union + 1e-7)


def soft_iou(predictions, labels):
    # http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
    predictions = tf.reshape(predictions, [-1])
    truth = tf.reshape(labels, [-1])
    intersection = tf.reduce_sum(predictions * truth)
    union = tf.reduce_sum(predictions + truth) - intersection
    union = tf.Print(union, [intersection, union])
    return 1 - intersection/(union + 1e-7)


def binary_crossentropy(logits, labels, pos_weight):
    return tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits, weights=pos_weight)


def dice(predictions, labels):
    masked = tf.cast(predictions > .5, tf.float32)
    return soft_dice(masked, labels)
