import glob
from shutil import copy2
import os
import numpy as np
import datetime
import tensorflow as tf


class Config(object):
    def __init__(self):
        self.experiment_name = 'Initial Test'
        self.results_path = self.__create_experiment(self.experiment_name)
        self.dtype = tf.float32
        tf.keras.backend.set_floatx('float32')
        self.mean = 33
        self.std = 36
        self.output_path = "logs"
        self.imagenet_mean = np.array([103.939, 116.779, 123.68])
        self.learning_rate = .001
        self.epochs = 100
        self.patience = 10
        self.slice_batch_size = 4
        self.freeze_resnet = False
        self.atlas_positive_freq = .0016779066593665076
        self.atlas_pos_weight = 50.
        self.brats_val_split = .15


    @staticmethod
    def __create_experiment(name):
        """
        Creates new folder in experiments/ directory and copies config

        :param name: short, preferably unique, description of experiment
        :return: path of the new directory
        """
        base = 'tmp_experiments/' + name.replace(' ', '_')
        unique = base

        # Experiments should have unique descriptions, but just in case...
        if os.path.isdir(unique):
            for i in range(1000):
                unique = base + '_' + str(i)
                if not os.path.isdir(unique):
                    break

        os.makedirs(unique)
        paths_to_copy = glob.iglob(r'*.py')
        for path in paths_to_copy:
            copy2(path, unique)

        # Delete old tensorboard files
        tf_event_files = glob.glob('logs/*')
        sorted(tf_event_files)
        for f in tf_event_files[:-1]:
            os.remove(f)

        # Record timestamp of run
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        open(os.path.join(unique, timestamp), 'a').close()
        return unique


def compute_mean(reader, ids):
    mean = 0
    std = 0
    for id in ids:
        d = reader.get_case(ids[0])
        label = d['labels']
        data = d['data']
        mean += np.mean(data)
        std += np.std(data)

    mean /= len(ids)
    std /= len(ids)
    print("Mean is %d, Std deviation is %d" % (mean, std))
    return mean, std
