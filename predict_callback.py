import os
import sys

import keras.callbacks
import numpy as np

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary

import eval


class PredictCallback(keras.callbacks.Callback):

    def __init__(self, reader, config, name):
        super().__init__()
        self.path = config.results_path + '/' + name + "/epoch"
        self.generator = reader

    def on_epoch_end(self, epoch, logs=None):
        originals, processed, targets = self.generator.get_sample_cases()
        predictions = self.model.predict(processed) # (10, 224, 224, 4)

        output_dir = self.path + str(epoch) + '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, original in enumerate(originals):

            ex = eval.visualize(original, predictions[i], targets[i])
            ex.save(output_dir + str(i) + '.jpg', 'JPEG')

#           crf_predictions = process_crf(processed[i, :, :, :], predictions[i, :, :, :]) # (224, 224, 4)
#           crf_ex = eval.visualize(original, crf_predictions, targets[i])
#           crf_ex.save(output_dir + str(i) + '_crf.jpg', 'JPEG')


def process_crf(inputs, predictions):
    # Based on http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/

    # Inputs is (H, W, C)
    # Predictions is (H, W, K)

    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the softmax_to_unary for more information
    predictions = predictions.transpose([2, 0, 1])
    unary = softmax_to_unary(predictions)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(inputs.shape[0], inputs.shape[1], 4)

    d.setUnaryEnergy(unary.reshape([4, -1]))

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
#   feats = create_pairwise_gaussian(sdims=(10, 10), shape=inputs.shape[:2])

#   d.addPairwiseEnergy(feats, compat=2,
#                       kernel=dcrf.DIAG_KERNEL,
#                       normalization=dcrf.NORMALIZE_SYMMETRIC)

#   # This creates the channel-dependent features --
#   # because the segmentation that we get from CNN are too coarse
#   # and we can use local channel features to refine them
#   # Not sure how applicable this is to MRIs; usually is color-dependent features
#   # Where the variation in color is more significant than the variation in the modalities
#   feats = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01, 0.01, 0.01, 0.01),
#                                     img=inputs, chdim=2)

#   d.addPairwiseEnergy(feats, compat=10,
#                       kernel=dcrf.DIAG_KERNEL,
#                       normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseGaussian(sxy=1, compat=4)

    Q = d.inference(5) # Number of inference steps

    Q = np.array(Q)

    res = Q.reshape(predictions.shape)
    res = res.transpose([1, 2, 0])

    return res
