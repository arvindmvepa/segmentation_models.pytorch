import os

from sklearn.model_selection import KFold

from torch.utils.data import DataLoader

import torch
import numpy as np
from numpy.random import RandomState
import segmentation_models_pytorch as smp
import json
import itertools
from .dataset import Dataset
from .losses import losses
from .metrics import metrics
from .optimizers import optimizers
from segmentation_models_pytorch import decoders
from .preprocessing import get_pos_wt, get_training_augmentation, get_validation_augmentation, get_preprocessing


def test_net(model_path, encoder='se_resnext50_32x4d', encoder_weights='imagenet', height=1024, width=1024,
             loss=('bce_lts', {}), data_dir='/root/data/vessels/test/images', seg_dir='/root/data/vessels/test/gt',
             save_dir='/root/output/vessels', save_preds=False, bs=1, test_metrics=(('accuracy', {}), ), device='cuda',
             cuda='0', *args, **kwargs):

    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if save_preds:
        save_preds_dir = os.path.join(save_dir, "preds")
        if not os.path.exists(save_preds_dir):
            os.makedirs(save_preds_dir)
    else:
        save_preds_dir = None


    model = torch.load(model_path)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    # create test dataset
    test_dataset = Dataset(
        data_dir,
        seg_dir,
        augmentation=get_validation_augmentation(height=height, width=width),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4)

    for i in range(len(test_metrics)):
        if test_metrics[i] != 'inf_time':
            test_metrics[i] = metrics[test_metrics[i][0]](**test_metrics[i][1])

    loss = losses[loss[0]](**loss[1])

    # evaluate model on test set
    test_epoch = smp.utils.test.TestEpoch(
        model=model,
        loss=loss,
        metrics=test_metrics,
        device=device,
        verbose=True,
        save_preds_dir=save_preds_dir
    )

    test_logs = test_epoch.run(test_dataloader)

    test_metrics = {test_metric: test_logs[test_metric]
                    for metric in metrics
                    for test_metric in test_logs.keys() if metric in test_metric}

    test_metrics.update({"model": model_path})
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as outfile:
        json.dump(test_metrics, outfile)


def val_net(model_path, encoder='se_resnext50_32x4d', encoder_weights='imagenet', height=1024, width=1024,
            loss=('bce_lts', {}), data_dir='/root/data/vessels/train/images', seg_dir='/root/data/vessels/train/gt',
            val_seg_dir=None, save_dir='/root/output/vessels', save_preds=False, bs=1, val_metrics=(('accuracy', {}), ),
            out_file=None, random_state=42, n_splits=10, fold=0, device='cuda', cuda='0', *args, **kwargs):

    os.environ['CUDA_VISIBLE_DEVICES'] = cuda

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if save_preds:
        save_preds_dir = os.path.join(save_dir, "preds")
        if not os.path.exists(save_preds_dir):
            os.makedirs(save_preds_dir)
    else:
        save_preds_dir = None

    val_metrics = list(val_metrics)

    if n_splits:
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        val_masks = sorted(list(os.listdir(val_seg_dir if val_seg_dir else seg_dir)))
        split_ids = list(kf.split(val_masks))[fold]
        val_ids = [val_masks[split_id] for split_id in split_ids[1]]
    else:
        val_ids = None

    model = torch.load(model_path)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    valid_dataset = Dataset(
        data_dir,
        seg_dir,
        augmentation=get_validation_augmentation(height=height, width=width),
        preprocessing=get_preprocessing(preprocessing_fn),
        ids=val_ids,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=4)

    loss = losses[loss[0]](**loss[1])
    for i in range(len(val_metrics)):
        if val_metrics[i] != 'inf_time':
            val_metrics[i] = metrics[val_metrics[i][0]](**val_metrics[i][1])

    valid_epoch = smp.utils.test.TestEpoch(
        model=model,
        loss=loss,
        metrics=val_metrics,
        device=device,
        verbose=True,
        save_preds_dir=save_preds_dir
    )

    valid_logs = valid_epoch.run(valid_loader)

    val_metrics = {val_metric: valid_logs[val_metric]
                   for metric in metrics
                   for val_metric in valid_logs.keys() if metric in val_metric}

    val_metrics.update({"model": model_path})
    if not out_file:
        out_file = "val" + os.path.basename(model_path)[:-4] + ".json"
    with open(os.path.join(save_dir, out_file), 'w') as outfile:
        json.dump(val_metrics, outfile)


def train_net(data_dir='/root/data/vessels/train/images', seg_dir='/root/data/vessels/train/gt',
              val_seg_dir=None, wt_masks_dir=None, train_sample_prop=1.0, train_sample_seed=1,
              save_dir='/root/exp', decoder="unet", encoder='se_resnext50_32x4d', encoder_weights='imagenet',
              activation='sigmoid', height=1024, width=1024, loss=('bce_lts', {}), pos_scale= None,
              optimizer=("adam", {"lr": 1e-4}), lr_schedule=((200, 1e-5), (400, 1e-6)), bs=8,
              train_metrics=(('accuracy', {}), ), val_metrics=(('accuracy', {}), ),
              best_metrics=(('accuracy_0.5', 0.0, [], True), ), best_thresh_metrics=(('accuracy', 0.0, True), ),
              last_metrics=('accuracy',), n_splits=10, fold=0, val_freq=5, checkpoint_freq=50, num_epochs=200,
              test_type="last", random_state=42, device='cuda', cuda='0', save_net=True):
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

    masks = sorted(list(os.listdir(seg_dir)))
    if train_sample_prop < 1.0:
        prng = RandomState(train_sample_seed)
        masks = prng.choice(masks, np.round(len(masks) * train_sample_prop))

    if n_splits:
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        val_masks = sorted(list(os.listdir(val_seg_dir))) if val_seg_dir else masks
        split_ids = list(kf.split(val_masks))[fold]
        val_ids = [val_masks[split_id] for split_id in split_ids[1]]
        train_ids = [mask for mask in masks if mask not in val_ids]
    else:
        train_ids = None
        val_ids = None

    train_dataset = Dataset(
        data_dir,
        seg_dir,
        wt_masks_dir=wt_masks_dir,
        augmentation=get_training_augmentation(height=height, width=width),
        preprocessing=get_preprocessing(preprocessing_fn),
        ids=train_ids,
    )
    valid_dataset = Dataset(
        data_dir,
        seg_dir,
        augmentation=get_validation_augmentation(height=height, width=width),
        preprocessing=get_preprocessing(preprocessing_fn),
        ids=val_ids,
    )

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    loss = losses[loss[0]](pos_weight=torch.FloatTensor([get_pos_wt(masks_fps=train_dataset.masks_fps, c=pos_scale)]),
                           reduction="none", **loss[1])

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