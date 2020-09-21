import torch
import nibabel as nib
import scipy.io as scio
from torch.utils.data import Dataset
from utils import data_augment as aug
import numpy as np
import re


brain_path = "E:/Y4/DT/data/resized_brain_64/"
affined_path = "E:/Y4/DT/data/affined_brain_64/"
label_path = "E:/Y4/DT/data/mat_64/"
template_path = "E:/Y4/DT/data/Resized64_MNI152_T1_1mm_brain.nii.gz"


# Prepare the dataset for the training
class DataProvider(Dataset):
    def __init__(self, input_set1, input_set2, input_set3=template_path, input_set4=None, norm=None, experiment=None):
        self.brain_img_set = input_set1
        self.label = input_set2
        self.template_path = input_set3
        self.affined_img_set = input_set4
        self.norm = norm
        self.experiment = experiment

    def __getitem__(self, index):
        brain_fn = self.brain_img_set[index]
        label_fn = self.label[index]

        brain_img = nib.load(brain_path + brain_fn)
        brain_img = brain_img.get_data()
        min_val = np.min(brain_img)
        max_val = np.max(brain_img)
        brain_norm = (brain_img - min_val) / (max_val - min_val)
        zero_norm = (0 - min_val) / (max_val - min_val)

        if self.affined_img_set is not None:
            affined_fn = self.affined_img_set[index]
            affined_img = nib.load(affined_path + affined_fn)
            affined_img = affined_img.get_data()
            min_val = np.min(affined_img)
            max_val = np.max(affined_img)
            affined_norm = (affined_img - min_val) / (max_val - min_val)
            affined_tensor = torch.from_numpy(affined_norm)

        template_img = nib.load(template_path)
        template_img = template_img.get_data()
        min_val = np.min(template_img)
        max_val = np.max(template_img)
        template_norm = (template_img - min_val) / (max_val - min_val)

        label = scio.loadmat(label_path + label_fn)
        label = label[list(label.keys())[3]]
        label = label[0:3,:]
        label = label.astype(np.float32)
        index = re.split('T1', brain_fn)[0]

        brain_tensor = torch.from_numpy(brain_norm)
        template_tensor = torch.from_numpy(template_norm)
        template_tensor = template_tensor.float()
        label_tensor = torch.from_numpy(label)

        if self.experiment is None:
            return brain_tensor, template_tensor, label_tensor, index

        if self.experiment == 1:
            return brain_tensor, template_tensor, label_tensor

        label[0][3] /= self.norm
        label[1][3] /= self.norm
        label[2][3] /= self.norm
        if self.experiment == 2 or self.experiment == 3:
            return brain_tensor, template_tensor, label_tensor

        if self.experiment == 4:
            return brain_tensor, template_tensor, affined_tensor, label_tensor

        if self.experiment == 5:
            Ta_label, Ta_img = aug.img_aug(brain_norm, self.norm, zero_norm)
            return brain_tensor, template_tensor, label_tensor, Ta_img, Ta_label

    def __len__(self):
        return len(self.brain_img_set)