import numpy as np
from tqdm import tqdm
from augmentation import preprocess


class EvalGenerator(object):
    'Generates data for Keras'

    def __init__(self, reader, list_ids, dim):
        'Initialization'
        self.reader = reader
        self.list_ids = list_ids

        height, width, channels = dim
        self.dim = (height, width, channels)
        self.n_channels = channels

        self.cases = []
        self.length = len(self.list_ids)

        for index, id in tqdm(enumerate(list_ids), total=len(list_ids), ncols=60):
            case = reader.get_case(id)
            data = np.stack([self.normalize(case['flair']),
                             self.normalize(case['t1']),
                             self.normalize(case['t1ce']),
                             self.normalize(case['t2'])], axis=-1)
            labels = case['labels']

            data = np.transpose(data, axes=[2, 0, 1, 3])
            labels = np.expand_dims(labels, 0)

            data, labels = preprocess(data, labels)
            labels = np.transpose(np.squeeze(labels, 0), [2, 0, 1])

            data = data.astype(np.float16)
            labels = labels.astype(np.uint8)
            self.cases.append((data, labels))

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
        return self.__data_generation(index)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.list_ids)

    def __data_generation(self, case_index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        return self.cases[case_index]
