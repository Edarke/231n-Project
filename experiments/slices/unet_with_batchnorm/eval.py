import numpy as np
import scipy.misc as misc
from PIL import Image

from read_data import ATLASReader

_true_color = np.array([[[255, 0, 0, 75]]])
_prediction_color = np.array([[[0, 0, 255, 75]]])
_background_color = np.array([[[0, 255, 0, 75]]])


def visualize(original, prediction, labels):
    """
    :param original: Original greyscale slice with dimension (h, w)
    :param prediction: Logits (or prediected labels) for each pixel in slice
    :param labels: Corresponding ground truth slice
    :return: An RGB PIL image that shows the overlap of our segmentation and the ground truth
    """
    probs = prediction.reshape([224, 224])
    prediction = prediction.reshape([224, 224, 1])
    labels = labels.reshape([224, 224, 1])

    background = (prediction == 0) * _background_color

    original = np.stack([original, original, original], axis=-1)

    filter = prediction * _prediction_color + labels * _true_color + background
    mask = misc.toimage(filter, cmin=0.0, cmax=255., mode='RGBA')
    original_image = misc.toimage(original, cmin=0., cmax=255.)
    original_image.paste(mask, box=None, mask=mask)

    probs *= 255
    probs = np.stack([probs, probs, probs], axis=-1)
    probs = misc.toimage(probs, cmin=0., cmax=255., mode='RGB')

    new_im = Image.new('RGB', (original_image.width*2, original_image.height))
    new_im.paste(original_image, (0, 0))
    new_im.paste(probs, (original_image.width, 0))
    return new_im




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
