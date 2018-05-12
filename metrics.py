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


def compute_mask(inputs):
    inputs = inputs[..., 0:1]
    background_signal = K.min(inputs, axis=[1, 2, 3], keepdims=True)
    return K.cast(inputs > background_signal, K.floatx())


def keras_masked_sparse_categorical_accuracy(mask):
    def accuracy(y_true, y_pred):
        y_true = K.squeeze(y_true, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)
        correct = K.sum(mask * (y_true == y_pred)) + 1e-7
        incorrect = K.sum(mask * (y_true != y_pred))
        return correct / (correct + incorrect)
    return accuracy



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
        y_true = K.squeeze(y_true, axis=-1)
        y_true = K.one_hot(K.cast(y_true, 'int64'), 4)  # (b, h, w, 4)

        # Ignore void class
        y_true = y_true[:, :, :, 1:]  # (b, h, w, 3)
        y_pred = y_pred[:, :, :, 1:]

        intersection = K.sum(y_true * y_pred, axis=[1, 2])
        union = K.sum(K.square(y_true), axis=[1, 2]) + K.sum(K.square(y_pred), axis=[1, 2])
        return K.mean((2. * intersection + smooth) / (union + smooth))

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


def hard_dice(y_true, y_pred):
    '''
    https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    return 2* (x intersection y) / (|x| + |y|)

    :param y_true: Sparse y labels
    :param y_pred:
    :return:
    '''
    smooth = 1e-5

    y_true = K.squeeze(y_true, axis=-1)
    y_true = K.one_hot(K.cast(y_true, 'int64'), 4)  # (b, h*w, 4)

    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), 4)

    # Ignore void class
    y_true = y_true[:, :, 1:]  # (b, h*w, 3)
    y_pred = y_pred[:, :, 1:]

    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(K.square(y_true), axis=[1, 2]) + K.sum(K.square(y_pred), axis=[1, 2])
    return K.mean((2. * intersection + smooth) / (union + smooth))


# TODO: Implement specificity and sensitivity metrics
