import os
import sys

import keras.callbacks


import eval


class PredictCallback(keras.callbacks.Callback):

    def __init__(self, reader, config, name):
        super().__init__()
        self.path = config.results_path + '/' + name + "/epoch"
        self.generator = reader

    def on_epoch_end(self, epoch, logs=None):
#       originals, processed, targets = self.generator.get_sample_cases()
#       predictions = self.model.predict(processed)

        output_dir = self.path + str(epoch) + '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

#       for i, original in enumerate(originals):
#           ex = eval.visualize(original, predictions[i], targets[i])
#           ex.save(output_dir + str(i) + '.jpg', 'JPEG')

#           crf_predictions = process_crf(processed[i, :, :, :], predictions[i, :, :, :]) # (224, 224, 4)
#           crf_ex = eval.visualize(original, crf_predictions, targets[i])
#           crf_ex.save(output_dir + str(i) + '_crf.jpg', 'JPEG')
