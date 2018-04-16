import glob
import os
import csv
import nrrd # For MCCAI
import numpy as np
import nibabel as nib # For ATLAS
from scipy.io import loadmat # for Cyprus
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from functools import reduce
from collections import defaultdict

datasets = {
    'mccai': 'datasets/MCCAI-2008',
    'cyprus': 'datasets/cyprus',
    'atlas': 'datasets/ATLAS_R1.1'
}

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


    def get_case(self, case_number):
        case_number = str(case_number)
        d = self.files[case_number]['directory']
        files = os.listdir(d)

        ret_files = {}
        for f in files:
            data = nib.load(os.path.join(d, f))
            ret_files[f] = data.get_data()

        return ret_files


class CyprusReader(object):
    def __init__(self):
        self.directory = datasets['cyprus']

    def get_files(self):
        pass
