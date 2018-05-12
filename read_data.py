import glob
import os
import csv
import nrrd  # For MCCAI
import random
import math
import unittest
import numpy as np
import nibabel as nib  # For ATLAS
from scipy.io import loadmat  # for Cyprus
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from functools import reduce
from collections import defaultdict
from abc import ABC, abstractmethod

datasets = {
    'mccai': 'datasets/MCCAI-2008',
    'cyprus': 'datasets/cyprus',
    'atlas': 'datasets/ATLAS_R1.1',
    'brats': 'datasets/MICCAI_BraTS_2018_Data_Training'
}


class AbstractReader(ABC):
    @abstractmethod
    def get_case_ids(self, val_p=0.15):
            return [None], [None]

    @abstractmethod
    def get_case(self, id):
        pass

    def get_mean_dev(self, val_p, modality):
        mean = 0
        var = 0
        training_set, _ = self.get_case_ids(val_p)
        for id in training_set:
            X = self.get_case(id)[modality]
            X = X[X > 0]
            mean += X.mean()
            var += X.var()
            print(X.mean(), X.var())
        return mean / len(training_set), np.sqrt(var / len(training_set))


class MCCAIReader(object):
    def __init__(self):
        self.directory = datasets['mccai']
        self.train_files = self.get_files(True)
        self.test_files = self.get_files(False)

    def get_files(self, train):
        train_test = 'train' if train else 'test1'

        # Get file names for all data
        dirs = glob.glob(os.path.join(self.directory, '*' + train_test + '_Part*'))

        cases = map(lambda t_dir: list(map(lambda d: os.path.join(t_dir, d),
                                           os.listdir(t_dir))), dirs)
        cases = reduce(lambda x, y: x + y, cases)

        files = map(lambda t_case: (os.path.basename(t_case), glob.glob(os.path.join(t_case, '*.nhdr'))), cases)

        files = dict(files)

        # Organize file names in dictionary

        for k, v in files.items():
            newdict = defaultdict(list)
            for f in v:
                if 'FLAIR' in f:
                    newdict['FLAIR'].append(f)
                elif 'T1' in f:
                    newdict['T1'].append(f)
                elif 'T2' in f:
                    newdict['T2'].append(f)
                elif 'lesion' in f:
                    newdict['lesion'].append(f)
            files[k] = newdict

        return files

    def get_case_ids(self):
        return list(self.train_files.keys())

    # Files: Dictionary generated by get_mccai_files
    # Case number: patient number
    # Modality: T1, T2, or FLAIR
    # Source: UNC, CHB
    def get_case(self, case_number, train_or_test='train', modality='T1', source='UNC'):
        files = self.train_files if train_or_test == 'train' else self.test_files
        train_or_test = 'train' if train_or_test == 'train' else 'test1'

        file_to_get = source + '_' + train_or_test + '_' + 'Case' + ('0' if case_number < 10 else '') + str(case_number)
        case = files[file_to_get]
        modality_file = case[modality][0]
        data, opts = nrrd.read(modality_file)
        return data


class ATLASReader(object):
    def __init__(self):
        self.directory = datasets['atlas']
        self.metadata_path = os.path.join(self.directory, 'ATLAS_Meta-Data_Release_1.1_standard_mni.csv')
        self.files = self.get_files()

    def get_files(self):
        cases = {}
        with open(self.metadata_path) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip headers
            for row in reader:
                key = row[1]
                site = row[0]
                session = row[2].strip()
                num_strokes_lh_cortical = row[3]
                num_strokes_lh_subcortical = row[4]
                num_strokes_rh_cortical = row[5]
                num_strokes_rh_subcortical = row[5]
                num_strokes_other = row[6]
                stroke_type = row[7]
                stroke_location = row[8]
                stroke_hemisphere = row[9]
                vascular_territory = row[10]
                pvh = row[11]
                dwmh = row[12]
                notes = row[13]

                values = {
                    'directory': os.path.join(self.directory, site, '0' + key, session),
                    'site': site,
                    'session': session,
                    'num_strokes_lh_cortical': num_strokes_lh_cortical,
                    'num_strokes_lh_subcortical': num_strokes_lh_subcortical,
                    'num_strokes_rh_cortical': num_strokes_rh_cortical,
                    'num_strokes_rh_subcortical': num_strokes_rh_subcortical,
                    'num_strokes_other': num_strokes_other,
                    'stroke_type': stroke_type,
                    'stroke_location': stroke_location,
                    'stroke_hemisphere': stroke_hemisphere,
                    'vascular_territory': vascular_territory,
                    'pvh': pvh,
                    'dwmh': dwmh,
                    'notes': notes
                }
                cases[key] = values
        return cases

    def get_case_ids(self):
        return list(self.files.keys())

    def get_case(self, case_number):
        case_number = str(case_number)
        d = self.files[case_number]['directory']
        files = os.listdir(d)

        ret_files = {}
        for f in files:
            if 'Lesion' in f:
                key = 'labels'
            else:
                key = 'data'
            data = nib.load(os.path.join(d, f))
            ret_files[key] = data.get_data()

        return ret_files


class BRATSReader(AbstractReader):
    def __init__(self, use_hgg=True, use_lgg=True):
        self.directory = datasets['brats']
        self.files = self.get_files(use_hgg, use_lgg)
        self.modalities = ['t1ce', 'flair', 't1', 't2']

    def get_dims(self):
        # Get dimensionality of first example
        data = self.get_case(self.get_case_ids()[0][0])
        return data['labels'].shape

    def get_case_ids(self, val_p=0.15, test_p=0.15):
        random.seed(101)
        all_files = sorted(list(self.files.keys()))

        validation_indices = random.sample(range(len(all_files)), math.floor(len(all_files) * val_p))
        validation_ids = [all_files[i] for i in sorted(validation_indices)]

        remaining_ids = [i for i in all_files if i not in set(validation_ids)]

        testing_indices = random.sample(range(len(remaining_ids)), math.floor(len(all_files) * test_p))
        testing_ids = [remaining_ids[i] for i in sorted(testing_indices)]

        training_ids = [i for i in remaining_ids if i not in set(testing_ids)]

        random.shuffle(training_ids)

        return training_ids, validation_ids, testing_ids

    def get_case(self, case_id):
        ret_files = {}

        path, file_names = self.files[case_id]
        full_paths = map(lambda fname: os.path.join(path, fname), file_names)

        for full_path in full_paths:
            for modality in self.modalities:
                if full_path.endswith(modality + '.nii'):
                    ret_files[modality] = nib.load(full_path).get_data()
                elif full_path.endswith('seg.nii'):
                    data = nib.load(full_path).get_data()
                    label = data
                    label[label == 4] = 3
                    ret_files['labels'] = label

        return ret_files

    def get_files(self, use_hgg, use_lgg):
        files = {}
        subdirs = []

        if use_hgg:
            subdirs.append('HGG')
        if use_lgg:
            subdirs.append('LGG')

        for subdir in subdirs:
            patient_dirs = list(os.walk(os.path.join(self.directory, subdir)))
            for patient_dir in patient_dirs[1:]:
                files[os.path.basename(patient_dir[0])] = (patient_dir[0], patient_dir[2])
        return files


class CyprusReader(object):
    def __init__(self):
        self.directory = datasets['cyprus']
        self.patient_ids = self.populate_ids()
        self.files = self.get_files()

    def populate_ids(self):
        self.patient_ids = [name for name in os.listdir(self.directory) if
                            os.path.isdir(os.path.join(self.directory, name))]

    def get_files(self):
        self.files = {}
        for p_id in self.patient_ids:
            initial_mri_dir = os.path.join(self.directory, p_ids, '1')
            secondary_mri_dir = os.path.join(self.directory, p_ids, '2')


class BRATSReaderTest(unittest.TestCase):
    def setUp(self):
        self.breader = BRATSReader(use_hgg=True, use_lgg=True)
        self.breader2 = BRATSReader(use_hgg=True, use_lgg=True)

        self.t,  self.val,  self.test  = self.breader.get_case_ids(val_p = 0.15, test_p = 0.15)
        self.t2, self.val2, self.test2 = self.breader.get_case_ids(val_p = 0.15, test_p = 0.15)

    def test_deterministic(self):
        self.assertEqual(self.t,    self.t2)
        self.assertEqual(self.val,  self.val2)
        self.assertEqual(self.test, self.test2)

    def test_mutually_exclusive(self):
        self.assertEqual(len(set(self.val).intersection( set(self.test))), 0)
        self.assertEqual(len(set(self.val).intersection( set(self.t))),    0)
        self.assertEqual(len(set(self.test).intersection(set(self.t))),    0)

    def test_train_val_equal(self):
        self.assertEqual(len(self.val), len(self.test)) 

