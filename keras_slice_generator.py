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
        self.dim = (height, width, channels)
        self.batch_size = batch
        self.n_channels = channels
        self.on_epoch_end()

        self.cases = []
        self.use_ram = config.use_ram
        if self.use_ram:
            self.list_ids = list(itertools.product(range(len(list_ids)), range(num_slices)))
            for index, id in enumerate(list_ids):
                case = reader.get_case(id)
                data = np.stack([self.normalize(case['flair']), self.normalize(case['t1']), self.normalize(case['t1ce']), self.normalize(case['t2'])], axis=-1)
                labels = case['labels']

                data = np.transpose(data, axes=[2, 0, 1, 3])
                labels = np.expand_dims(labels, 0)

                data, labels = preprocess(data, labels, config)
                labels = np.transpose(np.squeeze(labels, 0), [2, 0, 1])

                data = data.astype(np.float16)
                labels = labels.astype(np.uint8)
                self.cases.append((data, labels))
                print(index)

    def normalize(self, x):
        mask = x[x > 0]
        mean = mask.mean()
        std = mask.std()
        return (x - mean) / std

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
        y = np.empty(shape[:-1], dtype=np.int8)

        if self.use_ram:
            shape = (len(list_ids_temp), 224, 224, 4)
            X = np.empty(shape)
            y = np.empty(shape[:-1], dtype=np.int8)
            for i, (case_index, slice_index) in enumerate(list_ids_temp):
                data, label = self.cases[case_index]
                X[i] = data[slice_index]
                y[i] = label[slice_index]
            return X, np.expand_dims(y, -1)
        # print('Generating data for indices', list_IDs_temp)
        # Generate data
        for i, (patient_id, slice_index) in enumerate(list_ids_temp):
            # Store sample
            dic = self.reader.get_case(patient_id)

            X[i, :, :, 0] = self.normalize(dic['t1ce'])[:, :, slice_index]
            X[i, :, :, 1] = self.normalize(dic['t1'])[:, :, slice_index]
            X[i, :, :, 2] = self.normalize(dic['t2'])[:, :, slice_index]
            X[i, :, :, 3] = self.normalize(dic['flair'])[:, :, slice_index]

            # Store class
            y[i] = dic['labels'][:, :, slice_index]
        y = np.expand_dims(y, axis=-1)
        return preprocess(X, y, self.config)

    def get_sample_cases(self, num_samples=10):
        samples = self.list_ids[0:num_samples]
        processed, targets = self.__data_generation(samples)
        originals = processed[:, :, :, 2]
        return originals, processed, targets
