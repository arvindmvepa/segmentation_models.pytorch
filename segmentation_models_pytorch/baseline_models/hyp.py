from sklearn.model_selection import KFold
import os
import numpy as np
from .match import matched_filter
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm
from ..utils.dataset import Dataset
from sklearn.metrics import roc_auc_score


def hyp_search(data_dir='/root/data/vessels/train/images',
               seg_dir='/root/data/vessels/train/gt', num_searches=100, output_file="results.txt", n_splits=10,
               random_state=42):
    lengths = list(range(3, 13, 1))
    sigmas = list(range(6, 25, 1))
    thresholds = list(range(6, 36, 3))

    total_hp_combos = list(product(lengths, sigmas, thresholds))
    samp_hp_combos = sample(total_hp_combos, num_searches)

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    masks = sorted(list(os.listdir(seg_dir)))

    auc_score_max = 0
    length_max = 0
    sigma_max = 0
    threshold_max = 0

    for length, sigma, threshold in samp_hp_combos:
        auc_fold_scores = []
        for fold, (train_ids, val_ids) in enumerate(kf.split(masks)):
            train_ids = [masks[split_id] for split_id in train_ids]
            val_ids = [masks[split_id] for split_id in val_ids]

            train_dataset = Dataset(
                data_dir,
                seg_dir,
                ids=train_ids,
            )
            valid_dataset = Dataset(
                data_dir,
                seg_dir,
                ids=val_ids,
            )

            # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=12)
            val_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=12)

            with tqdm(val_loader,
                      desc="mf_search, fold: {} length: {}, sigma: {}, threshold: {}".format(fold,
                                                                                             length,
                                                                                             sigma,
                                                                                             threshold)) as iterator:
                predictions = []
                targets = []
                for img, target in iterator:
                    img = img.cpu().detach().numpy()[0, :, :, 0]
                    target = target.cpu().detach().numpy()[0, :, :, 0]
                    prediction = matched_filter(image=img, length=length, sigma=sigma / 10, coefficient=threshold / 10)
                    predictions.append(prediction)
                    targets.append(target)

                predictions = np.array(predictions)
                targets = np.array(targets)
                prediction_flat = predictions.flatten()
                target_flat = targets.flatten()

                auc_score = roc_auc_score(target_flat, prediction_flat)
                auc_fold_scores.append(auc_score)
                print("auc_score", auc_score)
                f = open(output_file, 'a')
                f.write("length: {}, sigma: {}, threshold: {}, fold: {}, auc_score: {}\n".format(length,
                                                                                                 sigma,
                                                                                                 threshold,
                                                                                                 fold,
                                                                                                 auc_score))
                f.close()
        mean_auc_score = np.mean(auc_fold_scores)
        print("mean auc_score", mean_auc_score)
        f = open(output_file, 'a')
        f.write("length: {}, sigma: {}, threshold: {}, mean_auc_score: {}\n".format(length, sigma, threshold,
                                                                                    mean_auc_score))
        f.close()
        if mean_auc_score > auc_score_max:
            auc_score_max = mean_auc_score
            length_max = length
            sigma_max = sigma
            threshold_max = threshold
    print("auc_score_max", type(auc_score_max), auc_score_max)
    print("length_max", type(length_max), length_max)
    print("sigma_max", type(sigma_max), sigma_max)
    print("threshold_max", type(threshold_max), threshold_max)
    return auc_score_max, length_max, sigma_max, threshold_max