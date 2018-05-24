import numpy as np
import scipy.misc as misc
from PIL import Image
import keras.utils
import keras.backend as K


from read_data import ATLASReader

_true_color = np.array([[[255, 0, 0, 75]]])
_prediction_color = np.array([[[0, 255, 0, 75]]])
_background_color = np.array([[[63, 127, 255, 100]]])

white = np.array([[[255, 255, 255, 100]]])
red = np.array([[[255, 0, 0, 100]]])
green = np.array([[[0, 255, 0, 100]]])
blue = np.array([[[0, 0, 255, 100]]])


def set_alpha(arr, alpha):
    arr[..., -1] = alpha
    return arr


# pred_to_color = {
#     0: white - white,
#     1: set_alpha(white - red, 100),
#     2: set_alpha(white - green, 100),
#     3: set_alpha(white - blue, 100)
# }

pred_to_color = {
    0: white - white,
    1: red,
    2: green,
    3: blue
}


def get_prediction_color(id):
    return np.array(pred_to_color[id])

# def get_truthy_color(id):
#     return np.array(truth_to_color[id])


colorize_prediction = np.vectorize(get_prediction_color, signature='()->(n)')
# colorize_labels = np.vectorize(get_truthy_color, signature='()->(n)')


def visualize(original, prediction, labels):
    """
    :param original: Original greyscale slice with dimension (h, w)
    :param prediction: Logits for each pixel in slice
    :param labels: Corresponding ground truth slice
    :return: An RGB PIL image that shows the overlap of our segmentation and the ground truth, and class probabilities
    """
    # prediction = np.round(np.clip(prediction, 0, 3))
    prediction = np.cumsum(prediction, axis=-1) > .5
    prediction = np.argmax(prediction, axis=-1)
    # prediction = prediction.reshape([224, 224]).astype(np.int32)
    labels = labels.reshape([224, 224])
    original = original.reshape([224, 224, 1])

    colored_preds = colorize_prediction(prediction)
    colored_labels = colorize_prediction(labels)
    background_mask = (original == original.min()) * _background_color

    prediction_mask = colored_preds + background_mask
    labels_mask = colored_labels + background_mask
    error_mask = (labels != prediction)[:, :, np.newaxis] * red + background_mask

    original = np.concatenate([original, original, original], axis=-1)
    original_image = misc.toimage(original)

    prediction_mask = misc.toimage(prediction_mask, cmin=0.0, cmax=255., mode='RGBA')
    labels_mask = misc.toimage(labels_mask, cmin=0.0, cmax=255., mode='RGBA')
    error_mask = misc.toimage(error_mask, cmin=0.0, cmax=255., mode='RGBA')

    joint_img = Image.new('RGB', (original_image.width * 3, original_image.height))
    joint_img.paste(original_image, box=(0, 0))
    joint_img.paste(original_image, box=(original_image.width, 0))
    joint_img.paste(original_image, box=(2 * original_image.width, 0))

    joint_img.paste(prediction_mask, box=(0, 0), mask=prediction_mask)
    joint_img.paste(labels_mask, box=(original_image.width, 0), mask=labels_mask)
    joint_img.paste(error_mask, box=(2 * original_image.width, 0), mask=error_mask)

    return joint_img


def crf(input, probs):
    pass


def np_dice_score(y_true, y_pred, category):
    def to_binary(y):
        y = y.reshape([y.shape[0], -1])  # (b, w*h)
        wt = y >= category
        return np.cast(wt, np.float32)

    y_pred = np.cumsum(y_pred, axis=-1)  # (b, h, w, c)
    y_pred = np.cast(y_pred >= .5, dtype=np.float32)  # (b, h, w, c)
    y_pred = np.argmax(y_pred, axis=-1)  # (b, h, w)

    smooth = 1e-8

    y_true = to_binary(y_true)  # (b, h*w)
    y_pred = to_binary(y_pred)  # (b, h*w)

    intersection = np.sum(np.multiply(y_true, y_pred), axis=1) + smooth
    union = np.sum(y_true, axis=1) + np.sum(y_pred, axis=1) + smooth
    return np.sum(2 * intersection / union, axis=0)


def evaluate(model, generator):
    scores = np.array([0, 0, 0])
    total = 0
    for input, label in generator:
        probs = model.predict(input)
        probs = crf(input, probs)
        total += label.shape[0]
        scores += [np_dice_score(label, probs, 1), np_dice_score(label, probs, 2), np_dice_score(label, probs, 3)]
    return scores / total



# For testing
if __name__ == '__main__':
    # Test case
    reader = ATLASReader()
    ids = reader.get_case_ids()
    case = reader.get_case(ids[0])
    original = case['data'][70]
    labels = case['labels'][70]
    pred = np.random.rand(*original.shape)
    import cv2

    orig = cv2.resize(np.random.rand(224 // 16, 224 // 16), dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
    preds = cv2.resize(np.random.rand(224 // 16, 224 // 16, 4), dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
    label = cv2.resize(np.random.randint(0, 4, (224 // 8, 224 // 8, 1)), dsize=(224, 224),
                       interpolation=cv2.INTER_NEAREST)

    tinted = visualize(original=orig, prediction=preds, labels=label)
    tinted.show()
