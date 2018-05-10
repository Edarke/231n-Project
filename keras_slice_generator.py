import keras
import numpy as np
import itertools

import sys

from main import preprocess

class SliceGenerator(keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, reader, num_slices, list_ids, dim, config, use_ram=True):
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
        self.use_ram = use_ram
        if use_ram:
            self.list_ids = list(itertools.product(range(len(list_ids)), range(num_slices)))
            for index, id in enumerate(list_ids):
                case = reader.get_case(id)
                data = np.stack([case['flair'], case['t1'], case['t1ce'], case['t2']], axis=-1)
                labels = case['labels']

                data = np.transpose(data, axes=[2, 0, 1, 3])
                labels = np.expand_dims(labels, 0)

                data, labels = preprocess(data, labels, config)
                labels = np.transpose(np.squeeze(labels, 0), [2, 0, 1])

                data = self.normalize(data)
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

            X[i, :, :, 0] = self.normalize(dic['t1ce'])
            X[i, :, :, 1] = self.normalize(dic['t1'])
            X[i, :, :, 2] = self.normalize(dic['t2'])
            X[i, :, :, 3] = self.normalize(dic['flair'])

            # Store class
            y[i] = dic['labels'][:, :, slice_index]
        y = np.expand_dims(y, axis=-1)
        return preprocess(X, y, self.config)

    def get_sample_cases(self, num_samples=10):
        originals = []
        samples = self.list_ids[0:num_samples]

        if self.use_ram:
            processed, targets = self.__data_generation(samples)
            originals = processed[:, :, :, 2]
        else:
            # TODO: Update this to use 4 channels
            for id, slice in samples:
                originals.append(np.expand_dims(self.reader.get_case(id)['t1ce'][:, :, slice], -1))
            processed, targets = self.__data_generation(samples)
            originals, _ = preprocess(np.array(originals), targets, self.config)
        return originals, processed, targets
