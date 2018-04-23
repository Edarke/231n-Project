import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
from read_data import ATLASReader


_true_color = np.array([[[255, 0, 0, 100]]])
_prediction_color = np.array([[[0, 0, 255, 100]]])

def visualize(original, prediction, labels):
    thresholded = (prediction > .5).astype(np.int32)
    thresholded = np.expand_dims(thresholded, -1)
    labels = np.expand_dims(labels, -1)
    original = np.stack([original, original, original], axis=-1)

    print(_true_color.shape, thresholded.shape)

    filter = thresholded * _prediction_color + labels * _true_color
    mask = misc.toimage(filter, cmin=0.0, cmax=255., mode='RGBA')
    original_image = misc.toimage(original, cmin=0., cmax=255.)
    original_image.paste(mask, box=None, mask=mask)
    return original_image






if __name__ == '__main__':
    reader = ATLASReader()
    ids = reader.get_case_ids()
    case = reader.get_case(ids[0])
    original = case['data'][70]
    labels = case['labels'][70]
    print(np.unique(np.ravel(labels)))
    pred = np.random.rand(*labels.shape)

    tinted = visualize(original=original, prediction=pred, labels=labels)
    tinted.show()