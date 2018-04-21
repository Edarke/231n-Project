import glob
from shutil import copy2
import os

import datetime


class Config(object):
    def __init__(self):
        self.experiment_name = 'Initial Test'
        self.results_path = self.__create_experiment(self.experiment_name)

    @staticmethod
    def __create_experiment(name):
        """
        Creates new folder in experiments/ directory and copies config

        :param name: short, preferably unique, description of experiment
        :return: path of the new directory
        """
        base = 'experiments/' + name.replace(' ', '_')
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

        # Record timestamp of run
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        open(os.path.join(unique, timestamp), 'a').close()
        return unique
