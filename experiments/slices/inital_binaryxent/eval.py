import numpy as np
import scipy.misc as misc
from PIL import Image

from read_data import ATLASReader

_true_color = np.array([[[255, 0, 0, 75]]])
_prediction_color = np.array([[[0, 255, 0, 75]]])
_background_color = np.array([[[0, 0, 255, 75]]])


def visualize(original, prediction, labels):
    """
    :param original: Original greyscale slice with dimension (h, w)
    :param prediction: Logits for each pixel in slice
    :param labels: Corresponding ground truth slice
    :return: An RGB PIL image that shows the overlap of our segmentation and the ground truth, and class probabilities
    """
    probs = prediction.reshape([224, 224])
    prediction = prediction.reshape([224, 224, 1])
    labels = labels.reshape([224, 224, 1])
    original = original.reshape([224, 224, 1])
    background_mask = original == original.min()

    original = np.concatenate([original, original, original], axis=-1)
    original_image = misc.toimage(original, cmin=0., cmax=255.)

    mask = prediction * _prediction_color + labels * _true_color + background_mask * _background_color
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
    pred = np.random.rand(*labels.shape)

    tinted = visualize(original=original, prediction=pred, labels=labels)
    tinted.show()
