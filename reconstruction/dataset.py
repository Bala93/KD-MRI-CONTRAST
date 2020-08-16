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

class SliceDataMulti(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor, mode): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 

        self.examples = []
        self.acc_factor = acc_factor 
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)

        self.flair_dir = os.path.join(root, 'mrbrain_flair', 'cartesian', mode,'acc_{}'.format(acc_factor))
        self.t1_dir    = os.path.join(root, 'mrbrain_t1', 'cartesian', mode, 'acc_{}'.format(acc_factor))

        files = glob.glob(os.path.join(self.flair_dir,'*.h5'))

        for file_path in sorted(files):
            with h5py.File(file_path,'r') as hf:
                #print (hf.keys())
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                fname = os.path.basename(file_path)
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i] 

        t1_path = os.path.join(self.t1_dir,fname)
        flair_path = os.path.join(self.flair_dir,fname)

        with h5py.File(flair_path, 'r') as data:

            flair_img  = data[self.key_img][:,:,slice]
            flair_kspace  = data[self.key_kspace][:,:,slice]
            flair_kspace = npComplexToTorch(flair_kspace)
            flair_target = data['volfs'][:,:,slice].astype(np.float64)

        with h5py.File(t1_path, 'r') as data:

            t1_img  = data[self.key_img][:,:,slice]
            t1_kspace  = data[self.key_kspace][:,:,slice]
            t1_kspace = npComplexToTorch(t1_kspace)
            t1_target = data['volfs'][:,:,slice].astype(np.float64)

            return torch.from_numpy(flair_img), flair_kspace, torch.from_numpy(t1_img), t1_kspace, torch.from_numpy(flair_target), torch.from_numpy(t1_target), fname, slice


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor, mode, contrast): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 

        self.examples = []
        self.acc_factor = acc_factor 
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)

        self.data_dir = os.path.join(root, 'mrbrain_{}'.format(contrast), 'cartesian', mode,'acc_{}'.format(acc_factor))

        files = glob.glob(os.path.join(self.data_dir,'*.h5'))

        for file_path in sorted(files):
            with h5py.File(file_path,'r') as hf:
                #print (hf.keys())
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                fname = os.path.basename(file_path)
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i] 

        data_path = os.path.join(self.data_dir,fname)

        with h5py.File(data_path, 'r') as data:

            img  = data[self.key_img][:,:,slice]
            kspace  = data[self.key_kspace][:,:,slice]
            kspace = npComplexToTorch(kspace)
            target = data['volfs'][:,:,slice].astype(np.float64)

            return torch.from_numpy(img), kspace, torch.from_numpy(target), fname, slice

