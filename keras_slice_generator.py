import keras
import numpy as np
import itertools

class SliceGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, reader, num_slices, list_ids, dim):
        'Initialization'
        self.reader = reader

        self.list_ids = list(itertools.product(list_ids, range(num_slices)))
        self.dim = dim
        self.batch_size = dim[0]
        self.n_channels = dim[-1]
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_ids_tmp = [self.list_ids[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_ids_tmp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_ids))
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty(self.dim)
        y = np.empty(self.dim[0:-1], dtype=np.int8)

        # Generate data
        for i, (patient_id, slice_index) in enumerate(list_IDs_temp):
            # Store sample
            dic = self.reader.get_case(patient_id)


            X[i, :, :, :] = np.expand_dims(dic['flair'][:, :, slice_index], axis=-1)

            # Store class
            y[i] = dic['labels'][:, :, slice_index]

        return X, y
