import os
import sys
from sklearn.model_selection import KFold

from torch.utils.data import DataLoader

import torch
import numpy as np
from numpy.random import RandomState
import segmentation_models_pytorch as smp
import json
import itertools
from .dataset import Dataset, InferenceDataset
from .losses import losses
from .metrics import metrics
from .optimizers import optimizers
from segmentation_models_pytorch import decoders
from .preprocessing import get_pos_wt, get_training_augmentation, get_validation_augmentation, get_preprocessing


def generate_predictions(model_dir, data_dir, save_data_dir, encoder='se_resnext50_32x4d', encoder_weights='imagenet',
                         bs=1, cuda='0', **kwargs):

    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(model_dir, f'best_model_{kwargs["best_metrics"][0][0]}.pth')

    model = torch.load(model_path)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    inf_dataset = InferenceDataset(
        data_dir,
        augmentation=None,
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    dataloader = DataLoader(inf_dataset, batch_size=bs, shuffle=False, num_workers=4)

    model.eval()
    with tqdm(dataloader, file=sys.stdout) as iterator:
        for img, file_path in iterator:
            img = img.to(self.device)
            with torch.no_grad():
                y_pred = model(img)
            y_pred = np.float32(y_pred.cpu().detach().numpy())
            np.save(os.path.join(save_data_dir, os.path.basename(file_path)), y_pred)



def train_fsl(train_sample_masks, data_dir, seg_dir, save_dir, decoder="unet", encoder='se_resnext50_32x4d',
              encoder_weights='imagenet', activation='sigmoid', height=256, width=256, loss=('bce_lts', {}),
              pos_scale= None, optimizer=("adam", {"lr": 1e-4}), lr_schedule=((200, 1e-5), (400, 1e-6)), bs=8,
              train_metrics=(('accuracy', {}), ), best_metrics=(('accuracy_0.5', 0.0, [], True), ),
              metric_freq=10, num_epochs=200, random_state=42, device='cuda', cuda='0', save_net=True, **kwargs):

    if len(train_sample_masks) == 0:
        raise ValueError("No training samples provided")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    json.dump(locals(), open(os.path.join(save_dir, "params.json"), 'w'))

    train_metrics = list(train_metrics)
    best_metrics = list(best_metrics)

    os.environ['CUDA_VISIBLE_DEVICES'] = cuda

    model = decoders[decoder](encoder_name=encoder,
                              encoder_weights=encoder_weights,
                              classes=1,
                              activation=activation)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    print("# of unique train images: {}".format(len(set(train_sample_masks))))

    train_dataset = Dataset(
        data_dir,
        seg_dir,
        ids=train_sample_masks,
        augmentation=get_training_augmentation(height=height, width=width),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=12)

    loss = losses[loss[0]](pos_weight=torch.FloatTensor([get_pos_wt(masks_fps=train_dataset.masks_fps, c=pos_scale)]),
                           reduction="none", **loss[1])

    for i in range(len(train_metrics)):
        if train_metrics[i] != 'inf_time':
            train_metrics[i] = metrics[train_metrics[i][0]](**train_metrics[i][1])

    optimizer = optimizers[optimizer[0]](params=model.parameters(), **optimizer[1])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=train_metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    for i in range(0, num_epochs):
        print('\nEpoch: {}'.format(i))
        train_logs == train_epoch.run(train_loader)
        cur_epoch = i + 1
        if cur_epoch % metric_freq == 0:
            for i in range(len(best_metrics)):
                metric, max_score, other_metrics, gt = best_metrics[i]
                max_score = save_best_checkpoint(model, metric, max_score, train_logs, cur_epoch, fold,
                                                 save_dir=save_dir, other_metrics=other_metrics, gt=gt,
                                                 save_net=save_net)
                best_metrics[i] = metric, max_score, other_metrics, gt

        for lr, epoch in lr_schedule:
            if i == epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                print('Changed Decoder learning rate to {}!'.format(str(lr)))


def save_best_checkpoint(model, metric, prev_max_score, valid_logs, cur_epoch, cur_fold, save_dir, other_metrics=None,
                         gt=True, save_net=True):
    if metric in valid_logs:
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
    else:
        raise ValueError('metric not found in valid_logs!')
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