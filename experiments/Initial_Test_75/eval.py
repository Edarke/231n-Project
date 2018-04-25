import numpy as np
import scipy.misc as misc
from read_data import ATLASReader

_true_color = np.array([[[255, 0, 0, 100]]])
_prediction_color = np.array([[[0, 0, 255, 100]]])


def visualize(original, prediction, labels):
    """
    :param original: Original greyscale slice with dimension (h, w)
    :param prediction: Logits (or prediected labels) for each pixel in slice
    :param labels: Corresponding ground truth slice
    :return: An RGB PIL image that shows the overlap of our segmentation and the ground truth
    """
    thresholded = (prediction > .5).astype(np.int32)
    thresholded = np.expand_dims(thresholded, -1)
    labels = np.expand_dims(labels, -1)
    original = np.stack([original, original, original], axis=-1)

    print(_true_color.shape, thresholded.shape)

    filter = thresholded * _prediction_color + labels * _true_color
    print(filter)
    mask = misc.toimage(filter, cmin=0.0, cmax=255., mode='RGBA')
    original_image = misc.toimage(original, cmin=0., cmax=255.)
    original_image.paste(mask, box=None, mask=mask)
    return original_image


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
