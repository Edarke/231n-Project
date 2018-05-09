import keras
import numpy as np
import itertools

import sys

from main import preprocess

class SliceGenerator(keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, reader, num_slices, list_ids, dim, config):
        'Initialization'
        self.reader = reader
        self.config = config
        self.list_ids = list(itertools.product(list_ids, range(num_slices)))
        self.list_ids = sorted(self.list_ids)

        batch, height, width, channels = dim
        self.dim = (height, width)
        self.batch_size = batch
        self.n_channels = channels
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        list_ids_tmp = self.list_ids[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(list_ids_tmp)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.list_ids)

    def __data_generation(self, list_ids_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        shape = (len(list_ids_temp), *self.dim)
        X = np.empty(shape)
        y = np.empty(shape, dtype=np.int8)

        # print('Generating data for indices', list_IDs_temp)
        # Generate data
        for i, (patient_id, slice_index) in enumerate(list_ids_temp):
            # Store sample
            dic = self.reader.get_case(patient_id)
            x = dic['t1ce']

            mask = x[x > 0]
            mean = mask.mean()
            std = mask.std()
            slice = x[:, :, slice_index]
            slice = (slice - mean) / std
            X[i, :, :] = slice

            # Store class
            y[i] = dic['labels'][:, :, slice_index]
        X = np.expand_dims(X, axis=-1)
        y = np.expand_dims(y, axis=-1)
        return preprocess(X, y, self.config)

    def get_sample_cases(self, num_samples=10):
        originals = []
        samples = self.list_ids[0:num_samples]
        for id, slice in samples:
            originals.append(np.expand_dims(self.reader.get_case(id)['t1ce'][:, :, slice], -1))
        processed, targets = self.__data_generation(samples)
        originals, _ = preprocess(np.array(originals), targets, self.config)
        return originals, processed, targets
