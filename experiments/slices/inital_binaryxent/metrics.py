import tensorflow as tf
import keras.backend as K

# def soft_dice(predictions, labels):
#     # 2TP / (T - TP + 2TP + P - TP)
#     predictions = tf.reshape(predictions, [-1])
#     truth = tf.reshape(labels, [-1])
#     intersection = 2*tf.reduce_sum(predictions * truth)
#     union = tf.reduce_sum(predictions + truth)
#     union = tf.Print(union, [intersection, union])
#     return 1 - intersection/(union + 1e-7)
from keras.layers.merge import _Merge


def soft_dice(y_pred, y_true):
    smooth = 1e-7
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f*y_true_f) + tf.reduce_sum(y_pred_f*y_pred_f) + smooth)


def soft_iou(predictions, labels, pos_weight):
    # http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
    predictions = tf.reshape(predictions, [-1])
    truth = tf.reshape(labels, [-1])
    intersection = tf.reduce_sum(predictions * truth)
    union = tf.reduce_sum(predictions + truth) - intersection
    union = tf.Print(union, [intersection, union])
    return 1 - intersection/(union + 1e-7)


def binary_crossentropy(logits, labels, pos_weight):
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits, pos_weight=pos_weight))


def dice(predictions, labels):
    masked = tf.cast(predictions > .5, tf.float32)
    return soft_dice(masked, labels)


def keras_mask_predictions(inputs_predictions_tuple):
    '''
    75% of the input is empty space. This multiplies the background by zero so it doesn't affect gradients.
    The background is composed of pixels with the minimal value in the slice. (This may differ between patients!)
    :param inputs_predictions_tuple:
    :return:
    '''
    inputs, predictions = inputs_predictions_tuple
    background_signal = K.min(inputs, axis=[1, 2, 3], keepdims=True)
    return K.cast(inputs > background_signal, K.floatx()) * predictions


# Call with something like:
# model.compile( ..., loss=keras_dice_coef_loss(input), ... )
def keras_dice_coef_loss(smooth=1):

    def keras_dice_coef(y_true, y_pred):
        '''
        https://github.com/keras-team/keras/issues/3611
        :param y_true:
        :param y_pred:
        :param smooth:
        :return:
        '''

        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
        return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    def keras_dice_coef_loss_fn(y_true, y_pred):
        '''
        The dice loss is minimized when the dice score is maximized.
        Outputs value in range (0, 1]
        https://github.com/keras-team/keras/issues/3611
        :param y_true:
        :param y_pred:
        :return:
        '''
        return 1 - keras_dice_coef(y_true, y_pred)

    return keras_dice_coef_loss_fn
