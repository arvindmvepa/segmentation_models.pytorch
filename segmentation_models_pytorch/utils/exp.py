import os

import cv2
from sklearn.model_selection import KFold

from torch.utils.data import DataLoader
import albumentations as albu

import torch
import numpy as np
import segmentation_models_pytorch as smp
import json
import itertools
from .dataset import Dataset
from .losses import losses
from .metrics import metrics
from .optimizers import optimizers
from segmentation_models_pytorch import decoders


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

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        albu.Resize(height=512, width=512, always_apply=True),
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


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(384, 480)
        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        albu.Resize(height=512, width=512, always_apply=True),
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


def test_net(model, encoder='se_resnext50_32x4d', encoder_weights='imagenet', loss=('bce_lts', {}),
             data_dir='/root/data/vessels/test/images', seg_dir='/root/data/vessels/test/gt',
             save_dir='/root/output/vessels', save_preds=False, bs=1, test_metrics=(('accuracy', {}), ),
             device='cuda', cuda='0'):

    os.environ['CUDA_VISIBLE_DEVICES'] = cuda

    if isinstance(model, str):
        model = torch.load(model)
    else:
        raise ValueError("Model string needed!")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    # create test dataset
    test_dataset = Dataset(
        data_dir,
        seg_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4)

    for i in range(len(test_metrics)):
        if test_metrics[i] != 'inf_time':
            test_metrics[i] = metrics[test_metrics[i][0]](**test_metrics[i][1])

    # evaluate model on test set
    test_epoch = smp.utils.test.TestEpoch(
        model=model,
        loss=loss,
        metrics=test_metrics,
        device=device,
        verbose=True,
        save_preds_dir=os.path.join(save_dir, "preds") if save_preds else None
    )

    test_logs = test_epoch.run(test_dataloader)

    test_metrics = {test_metric: test_logs[test_metric]
                    for metric in metrics
                    for test_metric in test_logs.keys() if metric in test_metric}

    test_metrics.update({"model": model})
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as outfile:
        json.dump(test_metrics, outfile)


def train_net(data_dir='/root/data/vessels/train/images', seg_dir='/root/data/vessels/train/gt',
              test_data_dir='/root/data/vessels/test/images', test_seg_dir='/root/data/vessels/test/gt',
              save_dir='/root/exp', decoder="unet", encoder='se_resnext50_32x4d', encoder_weights='imagenet',
              activation='sigmoid', loss=('bce_lts', {}), pos_scale= None, optimizer=("adam", {"lr": 1e-4}),
              lr_schedule=((200, 1e-5), (400, 1e-6)), bs=8, train_metrics=(('accuracy', {}), ),
              val_metrics=(('accuracy', {}), ), best_metrics=(('accuracy_0.5', 0.0, [], True), ),
              best_thresh_metrics=(('accuracy', 0.0, True), ), last_metrics=('accuracy',), n_splits=10, fold=0,
              val_freq=5, checkpoint_freq=50, num_epochs=200, test_type="last", random_state=42, device='cuda',
              cuda='0', save_net=True):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    json.dump(locals(), open(os.path.join(save_dir, "params.json"), 'w'))

    train_metrics = list(train_metrics)
    val_metrics = list(val_metrics)
    best_metrics = list(best_metrics)
    best_thresh_metrics = list(best_thresh_metrics)

    os.environ['CUDA_VISIBLE_DEVICES'] = cuda

    model = decoders[decoder](encoder_name=encoder,
                              encoder_weights=encoder_weights,
                              classes=1,
                              activation=activation)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    if n_splits:
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        masks = sorted(list(os.listdir(seg_dir)))
        split_ids = list(kf.split(masks))[fold]
        train_ids = [masks[split_id] for split_id in split_ids[0]]
        val_ids = [masks[split_id] for split_id in split_ids[1]]
    else:
        train_ids = None
        val_ids = None

    train_dataset = Dataset(
        data_dir,
        seg_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        ids=train_ids,
    )
    valid_dataset = Dataset(
        data_dir,
        seg_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        ids=val_ids,
    )

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    loss = losses[loss[0]](pos_weight=torch.FloatTensor([get_pos_wt(masks_fps=train_dataset.masks_fps, c=pos_scale)]),
                           **loss[1])

    for i in range(len(train_metrics)):
        if train_metrics[i] != 'inf_time':
            train_metrics[i] = metrics[train_metrics[i][0]](**train_metrics[i][1])

    for i in range(len(val_metrics)):
        if val_metrics[i] != 'inf_time':
            val_metrics[i] = metrics[val_metrics[i][0]](**val_metrics[i][1])

    optimizer = optimizers[optimizer[0]](params=model.parameters(), **optimizer[1])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=train_metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=val_metrics,
        device=device,
        verbose=True,
    )

    for i in range(0, num_epochs):
        print('\nEpoch: {}'.format(i))
        train_epoch.run(train_loader)
        cur_epoch = i + 1
        if cur_epoch % val_freq == 0:
            valid_logs = valid_epoch.run(valid_loader)
            if cur_epoch % checkpoint_freq == 0:
                save_last_checkpoint(model, last_metrics, valid_logs, cur_epoch, fold, save_dir=save_dir,
                                     save_net=save_net)

            for i in range(len(best_metrics)):
                metric, max_score, other_metrics, gt = best_metrics[i]
                max_score = save_best_checkpoint(model, metric, max_score, valid_logs, cur_epoch, fold,
                                                 save_dir=save_dir, other_metrics=other_metrics, gt=gt,
                                                 save_net=save_net)
                best_metrics[i] = metric, max_score, other_metrics, gt

            for i in range(len(best_thresh_metrics)):
                metric, max_score, gt = best_thresh_metrics[i]
                # bug, should use the `save_best_thresh_checkpoint`
                max_score = save_best_checkpoint(model, metric, max_score, valid_logs, cur_epoch, fold,
                                                 save_dir=save_dir, gt=gt,
                                                 save_net=save_net)
                best_thresh_metrics[i] = metric, max_score, gt

        for lr, epoch in lr_schedule:
            if i == epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                print('Changed Decoder learning rate to {}!'.format(str(lr)))


def save_best_checkpoint(model, metric, prev_max_score, valid_logs, cur_epoch, cur_fold, save_dir, other_metrics=None,
                         gt=True, save_net=True):
    if (metric in valid_logs):
        if ((valid_logs[metric] > prev_max_score) if gt
        else (valid_logs[metric] < prev_max_score)):
            max_score = valid_logs[metric]
            if save_net:
                torch.save(model, os.path.join(save_dir, 'best_model_' + metric + '.pth'))
            metrics = {metric: max_score, "epoch": cur_epoch, 'fold': cur_fold}
            if other_metrics:
                metrics.update({valid_metric: valid_logs[valid_metric]
                                for metric in other_metrics
                                for valid_metric in valid_logs.keys()
                                if metric in valid_metric})
            with open(os.path.join(save_dir, metric + '.json'), 'w') as outfile:
                json.dump(metrics, outfile)
            print(metric + ' Model saved!')
            return max_score
    return prev_max_score


def save_best_thresh_checkpoint(model, metric, prev_max_score, valid_logs, cur_epoch, cur_fold, save_dir, gt=True,
                                save_net=True):
    metrics = {valid_metric: valid_logs[valid_metric]
               for valid_metric in valid_logs.keys() if metric in valid_metric}
    if metrics:
        metric_vals = list(metrics.values())
        metric_val = max(metric_vals)

        if ((metric_val > prev_max_score) if gt
        else (metric_val < prev_max_score)):
            max_score = metric_val

            metric_names = list(metrics.keys())
            metric_name = metric_names[np.argmax(metric_vals)]

            if save_net:
                torch.save(model, os.path.join(save_dir, 'best_thresh_model_' + metric + '.pth'))
            metrics = {metric: str(max_score), "epoch": cur_epoch, 'fold': cur_fold,
                       "thresh": metric_name}
            with open(os.path.join(save_dir, 'thresh_' + metric + '.json'), 'w') as outfile:
                json.dump(metrics, outfile)
            print('thresh ' + metric + ' Model saved!')
            return max_score
    return prev_max_score


def save_last_checkpoint(model, metrics, valid_logs, cur_epoch, cur_fold, save_dir, save_net=True):
    if save_net:
        torch.save(model, os.path.join(save_dir, str(cur_epoch) + '.pth'))
        torch.save(model, os.path.join(save_dir, 'last.pth'))

    metrics = {valid_metric: valid_logs[valid_metric]
               for metric in metrics
               for valid_metric in valid_logs.keys() if metric in valid_metric}

    metrics.update({"epoch": cur_epoch, 'fold': cur_fold})
    with open(os.path.join(save_dir, str(cur_epoch) + '.json'), 'w') as outfile:
        json.dump(metrics, outfile)
    with open(os.path.join(save_dir, 'last.json'), 'w') as outfile:
        json.dump(metrics, outfile)
    print('Last Model saved!')


def grid_search(**kwargs):
    """
    This implements grid search for hyper-parameter search.
    Parameters
    ----------
    kwargs : dict
        A dictionary, where the key is the hyper-parameter, and the value is a list of possible hyper-parameter values
    Returns
    -------
    list
        A list of dicts in which each dict is a set of hyper-parameter choices for all the hyper-parameters
    Notes
    ------
    Lot of content taken from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists.
    """
    # hyper-parameter arg names
    keys = kwargs.keys()
    # hyper-parameter options
    vals = kwargs.values()

    searches = []
    # get cartesian product from all hyper-parameter options
    for instance in itertools.product(*vals):
        searches.append(dict(zip(keys, instance)))
    return searches


def remove_duplicates(lst):
    dup = set()
    new_lst = []
    for it in lst:
        json_it = json.dumps(it, sort_keys=True)
        if json_it not in dup:
            new_lst += [it]
            dup.add(json_it)
    return new_lst