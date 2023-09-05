import numpy as np
import os
import cv2
import h5py
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):

    def __init__(self, data_dir="ACDC/ACDC_training_slices", ids=None, num_classes=5, augmentation=None,
                 resize_width=224, resize_height=224, preprocessing=None):
        self.sample_list = []
        if not ids:
            self.all_slices = sorted(os.listdir(data_dir))
        else:
            self.all_slices = sorted([id for id in os.listdir(data_dir) if id in ids])
        self.sample_list = self.all_slices
        self.sample_list = [os.path.join(data_dir, im_path) for im_path in self.sample_list]
        self.class_values = [i for i in range(num_classes)]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.resize_width = resize_width
        self.resize_height = resize_height

        print("total {} samples".format(len(self.sample_list)))

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(case, 'r')
        # convert to 3 channels and float32
        image = h5f['image'][:]
        image = cv2.resize(image, (self.resize_width, self.resize_height))
        image = np.stack((image,) * 3, axis=-1).astype(np.float32)
        # convert to one-hot encoding
        label = h5f['label'][:]
        # TODO: labels need to be upsampled for correct evaluation
        label = cv2.resize(label, (self.resize_width, self.resize_height))
        label = [(label == v) for v in self.class_values]
        label = np.stack(label, axis=-1).astype(np.float32)


        # apply augmentations
        if self.augmentation:
            import sys
            sample = self.augmentation(image=image, mask=label)
            image, label = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=label)
            image, label = sample['image'], sample['mask']

        return image, label

    def __len__(self):
        return len(self.sample_list)


class InferenceDataset(Dataset):

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(case, 'r')
        # convert to 3 channels and float32
        image = h5f['image'][:]
        image = cv2.resize(image, (self.resize_width, self.resize_height))
        image = np.stack((image,) * 3, axis=-1).astype(np.float32)


        # apply augmentations
        if self.augmentation:
            import sys
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image, case

    def __len__(self):
        return len(self.sample_list)

