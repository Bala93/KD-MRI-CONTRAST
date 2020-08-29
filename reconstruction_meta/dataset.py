import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 
from utils import npComplexToTorch
import glob

class SliceData(Dataset):

    def __init__(self, root, acc_factor, mode): 

        self.examples = []
        self.acc_factor = acc_factor 
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)

        self.flair_dir = os.path.join(root, 'mrbrain_flair', 'acc_{}'.format(acc_factor), mode)
        self.t1_dir    = os.path.join(root, 'mrbrain_t1', 'acc_{}'.format(acc_factor), mode)

        flair_files = glob.glob(os.path.join(self.flair_dir,'*.h5'))
        t1_files = glob.glob(os.path.join(self.t1_dir,'*.h5'))

        # 1 -- flair, 2 -- t1
        for file_path in sorted(flair_files):
            with h5py.File(file_path,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(file_path, slice, 1) for slice in range(num_slices)]
        for file_path in sorted(t1_files):
            with h5py.File(file_path,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(file_path, slice, 2) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fpath, slice, context = self.examples[i] 

        with h5py.File(fpath, 'r') as data:

            img  = data[self.key_img][:,:,slice]
            kspace  = data[self.key_kspace][:,:,slice]
            kspace = npComplexToTorch(kspace)
            target = data['volfs'][:,:,slice].astype(np.float64)

            return torch.from_numpy(img), kspace, torch.from_numpy(target), fpath, slice, context
