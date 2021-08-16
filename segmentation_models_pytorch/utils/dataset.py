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
            omask_dir,
            extra_masks_dir=None,
            ids=None,
            model_pred_dirs=None,
            augmentation=None,
            preprocessing=None,
            val=False,
            test=False,
    ):
        if not ids:
            # Using numpy arrays
            self.ids = sorted(os.listdir(masks_dir))
        else:
            self.ids = ids
        self.ids = [id[:2] for id in self.ids]
        self.images_fps = [os.path.join(images_dir, image_id + "_test.tif") for image_id in self.ids] \
            if test else [os.path.join(images_dir, image_id + "_training.tif") for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id + "_manual1.gif.npy") for image_id in self.ids
                          if (image_id + "_manual1.gif.npy") in os.listdir(masks_dir)]
        self.omasks_fps = [os.path.join(omask_dir, image_id + "_test_mask.gif.npy") for image_id in self.ids if (image_id + "_test_mask.gif.npy") in os.listdir(omask_dir)] \
            if test else [os.path.join(omask_dir, image_id + "_training_mask.gif.npy") for image_id in self.ids if (image_id + "_training_mask.gif.npy") in os.listdir(omask_dir)]
        self.model_pred_dirs = model_pred_dirs
        if self.model_pred_dirs:
            self.pred_files = [id + ".npy" for id in self.ids]
        print("The number of masks are: {}. The masks are: {}".format(len(self.masks_fps), str(self.masks_fps)))
        # UPDATED: mask is equivalent 1
        self.class_values = [1]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.val = val

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_loc = self.masks_fps[i]
        omask_loc = self.omasks_fps[i]

        mask = np.load(mask_loc)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add a negative value for ignored pixels
        omask = (1 - np.load(omask_loc)) * -1
        omask = omask[:, :, np.newaxis]
        mask = mask + omask

        # apply augmentations
        if self.augmentation:
            """
            if not self.val:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            else:
                sample = self.augmentation(image=image)
                image = sample['image']
            """
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.model_pred_dirs:
            pred_file = self.pred_files[i]
            all_model_preds = [np.load(os.path.join(model_dir, "preds", pred_file)) for model_dir in self.model_pred_dirs]
            return np.mean(all_model_preds, axis=0), mask

        return image, mask

    def __len__(self):
        return len(self.ids)

