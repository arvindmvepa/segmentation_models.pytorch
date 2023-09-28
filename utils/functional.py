import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold == "argmax":
        indices = torch.argmax(x, dim=1, keepdim=True).type(torch.int64)
        one_hot = torch.zeros_like(x)
        one_hot.scatter_(1, indices, 1)
        return one_hot
    elif threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


jaccard = iou


def f_score(pr, gt, class_val=None, beta=1, eps=1e-7, threshold=None, accum=None, weighted=False, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """
    pr = _threshold(pr, threshold=threshold)

    if threshold is "argmax"  and class_val is None and (accum=="avg"):
        num_classes = pr.shape[1]
        score_ = 0.0
        for class_ in range(num_classes):
            if class_ == 0:
                continue
            pr_ = pr[:, class_, :, :]
            gt_ = gt[:, class_, :, :]
            tp = torch.sum(gt_ * pr_)
            fp = torch.sum(pr_) - tp
            fn = torch.sum(gt_) - tp
            score = ((1 + beta ** 2) * tp + eps) \
                    / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
            score_ += score
        return score_ / num_classes
    elif threshold is None and class_val is None and weighted:
        weights = 1 / (torch.sum(gt, dim=(0, 2, 3), keepdim=True)**2)
        tp = torch.sum(weights * gt * pr)
        fp = torch.sum(weights * pr) - tp
        fn = torch.sum(weights * gt) - tp
    elif threshold is None and class_val is None:
        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp
    elif threshold is "argmax" and class_val is not None:
        pr = pr[:, class_val, :, :]
        gt = gt[:, class_val, :, :]

        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp
    elif threshold is "argmax" and class_val is None:
        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp
    else:
        raise ValueError(f"Invalid threshold {threshold} and class_val {class_val} combination")

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


def auc_roc(pr, gt, ignore_channels=None):
    """Calculate auc_roc score between ground truth and prediction probs
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
    Returns:
        float: auc_roc score
    """
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    pr, gt = pr.cpu().detach().numpy().flatten(), np.rint(gt.cpu().detach().numpy().flatten())
    return torch.tensor(roc_auc_score(gt, pr))


def ap(pr, gt, ignore_channels=None):
    """Calculate ap score between ground truth and prediction probs
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
    Returns:
        float: auc_roc score
    """
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    pr, gt = pr.cpu().detach().numpy().flatten(), np.rint(gt.cpu().detach().numpy().flatten())
    return torch.tensor(average_precision_score(gt, pr))


def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp_tn = torch.sum(gt * pr) + torch.sum((1-gt) * (1-pr))
    score = tp_tn / gt.view(-1).shape[0]
    return score


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score
