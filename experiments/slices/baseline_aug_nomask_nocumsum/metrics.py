import tensorflow as tf
import keras.backend as K


from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary


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
def keras_dice_coef_loss(smooth=1e-8):

    def keras_dice_coef(y_true, y_pred):
        '''
        https://github.com/keras-team/keras/issues/3611
        :param y_true:
        :param y_pred:
        :param smooth:
        :return:
        '''
        y_true = K.squeeze(y_true, axis=-1)
        y_true = K.one_hot(K.cast(y_true, 'int32'), 4)  # (b, h, w, 4)

        # Ignore void class
        y_true = y_true[..., 1:]  # (b, h, w, 3)
        y_pred = y_pred[..., 1:]  # (b, h, w, 3)

        y_true = K.reshape(y_true, (-1, 3))  # (n, 3)
        y_pred = K.reshape(y_pred, (-1, 3))  # (n, 3)

        y2 = y_true[:, 2]
        y1 = y_true[:, 1] + y2
        y0 = y_true[:, 0] + y1
        y_true = tf.stack([y0, y1, y2], axis=-1)

        y2 = y_pred[:, 2]
        y1 = y_pred[:, 1]
        y0 = y_pred[:, 0]
        y_pred = tf.stack([y0, y1, y2], axis=-1)

        intersection = K.sum(y_true * y_pred, axis=0)  # (3)
        union = K.sum(K.square(y_true), axis=0) + K.sum(K.square(y_pred), axis=0)  # (3)
        return K.mean((2. * intersection + smooth) / (union + smooth))  # Mean dice loss
        return s

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


def category_dice_score(category):

    def hard_dice(y_true, y_pred):
        '''
        https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

        return 2* (x intersection y) / (|x| + |y|)

        :param y_true: Sparse y labels
        :param y_pred:
        :return:
        '''
        def to_binary(y):
            y = tf.reshape(y, [-1])
            wt = y >= category
            return tf.cast(wt, tf.float32)

        #y_pred = tf.cumsum(y_pred, axis=-1, exclusive=False, reverse=False)
        #y_pred = tf.cast(y_pred >= .5, dtype=tf.float32)
        y_pred = tf.argmax(y_pred, axis=-1)

        smooth = 1e-8
        y_true = to_binary(y_true)  # (3, b*h*w)
        y_pred = to_binary(y_pred)  # (3, b*h*w)

        intersection = 2 * tf.reduce_sum(y_true * y_pred) + smooth
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
        return intersection / union
    return hard_dice


wt_dice = category_dice_score(1)
wt_dice.__name__ = 'wt_dice'
tc_dice = category_dice_score(2)
tc_dice.__name__ = 'tc_dice'
et_dice = category_dice_score(3)
et_dice.__name__ = 'et_dice'
