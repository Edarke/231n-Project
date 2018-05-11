import numpy as np
import scipy.misc as misc
from PIL import Image

from read_data import ATLASReader

_true_color = np.array([[[255, 0, 0, 75]]])
_prediction_color = np.array([[[0, 255, 0, 75]]])
_background_color = np.array([[[0, 255, 255]]])

white = np.array([[[255, 255, 255]]])
red = np.array([[[255, 0, 0]]])
green = np.array([[[0, 255, 0]]])
blue = np.array([[[0, 0, 255]]])

pred_to_color = {
    0: white - white,
    1: white - red,
    2: white - green,
    3: white - blue
}

truth_to_color = {
    0: white - white,
    1: red,
    2: green,
    3: blue
}


def get_prediction_color(id):
    return np.array(pred_to_color[id])


def get_truthy_color(id):
    return np.array(truth_to_color[id])


colorize_prediction = np.vectorize(get_prediction_color, signature='()->(n)')
colorize_labels = np.vectorize(get_truthy_color, signature='()->(n)')


def visualize(original, prediction, labels):
    """
    :param original: Original greyscale slice with dimension (h, w)
    :param prediction: Logits for each pixel in slice
    :param labels: Corresponding ground truth slice
    :return: An RGB PIL image that shows the overlap of our segmentation and the ground truth, and class probabilities
    """
    prediction = prediction.reshape([224, 224, 4])
    probs = np.max(prediction, axis=-1)
    prediction = np.argmax(prediction, axis=-1)
    labels = labels.reshape([224, 224])
    original = original.reshape([224, 224, 1])

    colored_preds = colorize_prediction(prediction)
    colored_labels = colorize_labels(labels)
    background_mask = original == original.min() * _background_color

    mask = colored_preds + colored_labels + background_mask

    mask = mask / (np.max(mask, axis=-1, keepdims=True) + 1e-7) * 255
    alpha_channel = np.full((224, 224, 1), fill_value=100)
    mask = np.append(mask, alpha_channel, axis=-1)

    original = np.concatenate([original, original, original], axis=-1)
    original_image = misc.toimage(original)

    mask_img = misc.toimage(mask, cmin=0.0, cmax=255., mode='RGBA')
    original_image.paste(mask_img, box=None, mask=mask_img)

    probs = misc.toimage(probs, cmin=0., cmax=1., mode='L')
    joint_img = Image.new('RGB', (original_image.width * 2, original_image.height))
    joint_img.paste(original_image, box=(0, 0))
    joint_img.paste(probs, box=(original_image.width, 0))
    return joint_img


# For testing
if __name__ == '__main__':
    # Test case
    reader = ATLASReader()
    ids = reader.get_case_ids()
    case = reader.get_case(ids[0])
    original = case['data'][70]
    labels = case['labels'][70]
    print(np.unique(np.ravel(labels)))
    pred = np.random.rand(*original.shape)

    tinted = visualize(original=np.random.randn(224, 224, 1), prediction=np.random.rand(224, 224, 4), labels=np.random.randint(0, 4, (224, 224)))
    tinted.show()
