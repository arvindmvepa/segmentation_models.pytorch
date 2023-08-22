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


def get_training_augmentation(height=1024, width=1024):
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
        albu.Resize(height=height, width=width, always_apply=True),
        # albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
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

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
