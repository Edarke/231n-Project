import keras
import numpy as np
import itertools
from tqdm import tqdm
from augmentation import preprocess
import random


class SliceGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, reader, num_slices, list_ids, dim, config, augmentor, use_all_cross_sections):
        'Initialization'
        self.augmentor = augmentor
        self.reader = reader
        self.list_ids = list(itertools.product(list_ids, range(num_slices)))
        self.list_ids = sorted(self.list_ids)
        self.use_all_cross_sections = use_all_cross_sections

        batch, height, width, channels = dim
        self.dim = (height, width, channels)
        self.batch_size = batch
        self.n_channels = channels

        self.cases = []
        self.use_ram = config.use_ram
        self.length = int(np.ceil(len(self.list_ids) / self.batch_size))
        if self.use_ram:
            self.list_ids = list(itertools.product(range(len(list_ids)), range(num_slices)))

            for index, id in tqdm(enumerate(list_ids), total=len(list_ids), ncols=60):
                case = reader.get_case(id)
                data = np.stack(
                    [self.normalize(case['flair']), self.normalize(case['t1']), self.normalize(case['t1ce']),
                     self.normalize(case['t2'])], axis=-1)
                labels = case['labels']

                data = np.transpose(data, axes=[2, 0, 1, 3])
                labels = np.expand_dims(labels, 0)

                data, labels = preprocess(data, labels)
                labels = np.transpose(np.squeeze(labels, 0), [2, 0, 1])

                data = data.astype(np.float16)
                labels = labels.astype(np.uint8)
                self.cases.append((data, labels))
        self.on_epoch_end()

    def normalize(self, x):
        mask = x[x > 0]
        mean = mask.mean()
        std = mask.std()
        return (x - mean) / std

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.length

    def __getitem__(self, index):
        'Generate one batch of data'
        if index >= self.length:
            raise IndexError()
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

                if (self.use_all_cross_sections):
                    rand_num = random.random()
                    if (rand_num < 0.33):
                        slice = data[slice_index]
                        label = label[slice_index]
                    elif (rand_num < 0.66):
                        data = np.transpose(data, axes=[1, 0, 2, 3])
                        label = np.transpose(label, axes=[1, 0, 2])

                        slice = data[slice_index + 34]
                        label = label[slice_index + 34]
                        slice, label = preprocess(slice, label)
                    else:
                        data = np.transpose(data, axes=[2, 0, 1, 3])
                        label = np.transpose(label, axes=[2, 0, 1])

                        slice = data[slice_index + 34]
                        label = label[slice_index + 34]
                        slice, label = preprocess(slice, label)
                else:
                    slice = data[slice_index]
                    label = label[slice_index]

                X[i], y[i] = self.augmentor(slice, label)
            return X, np.expand_dims(y, -1)
        # print('Generating data for indices', list_IDs_temp)
        # Generate data
        for i, (patient_id, slice_index) in enumerate(list_ids_temp):
            # Store sample
            dic = self.reader.get_case(patient_id)

            slice = np.empty(self.dim)
            slice[:, :, 0] = self.normalize(dic['t1ce'])[:, :, slice_index]
            slice[:, :, 1] = self.normalize(dic['t1'])[:, :, slice_index]
            slice[:, :, 2] = self.normalize(dic['t2'])[:, :, slice_index]
            slice[:, :, 3] = self.normalize(dic['flair'])[:, :, slice_index]

            X[i], y[i] = (slice, dic['labels'][:, :, slice_index])

        y = np.expand_dims(y, axis=-1)
        X, y = preprocess(X, y)
        y = np.squeeze(y, axis=-1)
        for i in range(X.shape[0]):
            X[i], y[i] = self.augmentor(X[i], y[i])

    def get_sample_cases(self, num_samples=10):
        samples = self.list_ids[0:num_samples]
        processed, targets = self.__data_generation(samples)
        originals = processed[:, :, :, 2]
        return originals, processed, targets
