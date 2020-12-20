import numpy as np
import os
import cv2
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ["vessel"]

    def __init__(
            self,
            images_dir,
            masks_dir,
            wt_masks_dir=None,
            ids=None,
            augmentation=None,
            preprocessing=None,
    ):
        if not ids:
            # Using numpy arrays
            self.ids = sorted(os.listdir(masks_dir))
        else:
            self.ids = ids

        self.ids = [id[:-4] for id in self.ids]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id + ".npy") for image_id in self.ids]

        # add weighted elements
        if not wt_masks_dir:
            wt_masks_fps = self.masks_fps
            wt = 1
        elif len(wt_masks_dir) == 2:
            wt_masks_fps = sorted(os.listdir(wt_masks_dir[0]))
            wt = wt_masks_dir[1]
        else:
            raise ValueError("Wrong arg for wt_dir")
        self.wt_masks_fps = []
        for i in range(len(self.masks_fps)):
            mask_fps = self.masks_fps[i]
            if mask_fps in wt_masks_fps:
                self.wt_masks_fps.append((self.masks_fps[i], wt))
            else:
                self.wt_masks_fps.append((self.masks_fps[i], 1))

        # UPDATED: mask is equivalent 1
        self.class_values = [1]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_loc, wt = self.self.wt_masks_fps[i]

        mask = np.load(mask_loc)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, wt

    def __len__(self):
        return len(self.ids)
