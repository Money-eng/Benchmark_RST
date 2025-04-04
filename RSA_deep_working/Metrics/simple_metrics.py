from skimage.measure import label, euler_number
from skimage.metrics import adapted_rand_error
from sklearn.metrics import adjusted_rand_score, mutual_info_score
from sklearn.metrics.cluster import entropy
from skimage.morphology import skeletonize
import torchmetrics.functional as FMF
import torchmetrics.functional.segmentation as FMS
import numpy as np


def all_metrics():
    return [dice, f1_score, iou, pixel_accuracy, precision, recall, specificity,
            connectivity_metric, ARI_index, VI_index, ARE_error,
            betti_0_difference, euler_charac_difference]

# ----------------------------
# Standard segmentation metrics
# ----------------------------


def dice(prediction, mask, time=0, mtg=None):
    prediction = prediction.int()
    mask = mask.int()
    return FMS.dice_score(prediction, mask, num_classes=2, average='micro').mean().item()


def f1_score(prediction, mask, time=0, mtg=None):
    prediction = prediction.int()
    mask = mask.int()
    return FMF.f1_score(prediction, mask, average='micro', task='binary').mean().item()


def iou(prediction, mask, time=0, mtg=None):
    prediction = prediction.int()
    mask = mask.int()
    return FMF.jaccard_index(prediction, mask, task='binary').mean().item()


def pixel_accuracy(prediction, mask, time=0, mtg=None):
    prediction = prediction.int()
    mask = mask.int()
    return FMF.accuracy(prediction, mask, task='binary').mean().item()


def precision(prediction, mask, time=0, mtg=None):
    prediction = prediction.int()
    mask = mask.int()
    return FMF.precision(prediction, mask, task='binary').mean().item()


def recall(prediction, mask, time=0, mtg=None):
    prediction = prediction.int()
    mask = mask.int()
    return FMF.recall(prediction, mask, task='binary').mean().item()


def specificity(prediction, mask, time=0, mtg=None):
    prediction = prediction.int()
    mask = mask.int()
    _, fp, tn, _, _ = FMF.stat_scores(prediction, mask, task='binary')
    spec = (tn.float()) / (tn.float() + fp.float())
    return spec.mean().item()

# ----------------------------
# Topology-aware segmentation metrics
# ----------------------------


def connectivity_metric(prediction, mask, time=0, mtg=None):
    prediction = prediction.int().cpu().numpy().astype(np.uint8)
    mask = mask.int().cpu().numpy().astype(np.uint8)
    conn_scores = []
    for pred_img, mask_img in zip(prediction, mask):
        num_pred = label(pred_img).max()
        num_mask = label(mask_img).max()
        denominator = max(num_pred, num_mask)
        if denominator == 0:
            conn_score = 1.0 if num_pred == num_mask else 0.0
        else:
            conn_score = 1 - abs(num_pred - num_mask) / denominator
        conn_scores.append(conn_score)
    return np.mean(conn_scores).item()


def ARI_index(prediction, mask, time=0, mtg=None):
    prediction = prediction.int().cpu().numpy()
    mask = mask.int().cpu().numpy()
    return adjusted_rand_score(mask.flatten(), prediction.flatten())


def ARE_error(prediction, mask, time=0, mtg=None):
    prediction = prediction.int().cpu().numpy()
    mask = mask.int().cpu().numpy()
    try:
        are, _, _ = adapted_rand_error(mask, prediction)
    # if division by zero occurs
    except ZeroDivisionError or RuntimeError or RuntimeWarning:
        are = 0.0
    return are.item()


def VI_index(prediction, mask, time=0, mtg=None):
    prediction = prediction.int().cpu().numpy()
    mask = mask.int().cpu().numpy()
    H_mask = entropy(mask.flatten())
    H_pred = entropy(prediction.flatten())
    MI = mutual_info_score(mask.flatten(), prediction.flatten())
    VI = H_mask + H_pred - 2 * MI
    return VI


def betti_0_difference(prediction, mask, time=0, mtg=None):
    prediction = prediction.int().cpu().numpy().astype(np.uint8)
    mask = mask.int().cpu().numpy().astype(np.uint8)
    scores = []
    for pred_img, mask_img in zip(prediction, mask):
        num_pred = label(pred_img).max()
        num_mask = label(mask_img).max()
        # ratio
        scores.append(abs(num_pred - num_mask) /
                      (num_pred + num_mask))  # normalized
    return np.mean(scores).item()


def euler_charac_difference(prediction, mask, time=0, mtg=None):
    prediction = prediction.int().cpu().numpy().astype(np.uint8)
    mask = mask.int().cpu().numpy().astype(np.uint8)
    scores = []
    for pred_img, mask_img in zip(prediction, mask):
        euler_pred = euler_number(pred_img, connectivity=1)
        euler_mask = euler_number(mask_img, connectivity=1)
        scores.append(abs(euler_pred - euler_mask) /
                      (euler_pred + euler_mask))  # normalized
    return np.mean(scores).item()
