import albumentations as albu
import numpy as np


def get_neg_pos_ratio(masks_fps):
    masks_flat = np.concatenate([np.load(mask_fp)
                                 for mask_fp in masks_fps]).flatten()
    num_pos = float(np.sum(masks_flat))
    total = float(len(masks_flat))
    num_neg = total - num_pos
    return num_neg/num_pos


def get_pos_wt(masks_fps, c=1.0):
    if c:
        neg_pos_ratio = get_neg_pos_ratio(masks_fps)
        return c * neg_pos_ratio
    else:
        return 1.0


def get_training_augmentation():
    train_transform = [
        albu.Affine(rotate=10, translate_percent=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        albu.HueSaturationValue(),
        albu.ElasticTransform()
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(height=1024, width=1024):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(384, 480)
        albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
        albu.Resize(height=height, width=width, always_apply=True),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    x_ = x.transpose(2, 0, 1).astype('float32')
    return x_


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    if preprocessing_fn is not None:
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
    else:
        _transform = [
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
    return albu.Compose(_transform)
