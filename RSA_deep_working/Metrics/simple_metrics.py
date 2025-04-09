from skimage.measure import label, euler_number
from skimage.metrics import adapted_rand_error
from sklearn.metrics import adjusted_rand_score, mutual_info_score
from sklearn.metrics.cluster import entropy
import torchmetrics.functional as FMF
import torchmetrics.functional.segmentation as FMS
import numpy as np
import functools


def all_metrics():
    return [dice, f1_score, iou, pixel_accuracy, precision, recall, specificity,
            connectivity_metric, ARI_index, VI_index, ARE_error,
            betti_0_difference, euler_charac_difference]


# =============================================================================
# Décorateur pour standardiser les entrées des métriques
# =============================================================================
def standardize_metric(func):
    @functools.wraps(func)
    def wrapper(prediction, mask, time=0, mtg=None):
        # Conversion en int
        pred = prediction.int()
        msk = mask.int()
        # Suppression de la dimension de canal unique (si besoin)
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if msk.dim() == 4 and msk.size(1) == 1:
            msk = msk.squeeze(1)
        return func(pred, msk, time, mtg)
    return wrapper

def standardize_float_metric(func):
    @functools.wraps(func)
    def wrapper(prediction, mask, time=0, mtg=None):
        # Conversion en float (plutôt qu'en int)
        pred = prediction.float()
        msk = mask.float()
        # Supprimer la dimension de canal unique si présente
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if msk.dim() == 4 and msk.size(1) == 1:
            msk = msk.squeeze(1)
        return func(pred, msk, time, mtg)
    return wrapper

# =============================================================================
# Définition des métriques standards
# =============================================================================

@standardize_metric
def dice(prediction, mask, time=0, mtg=None):
    return FMS.dice_score(prediction, mask, num_classes=2, average='micro').mean().item()

@standardize_metric
def f1_score(prediction, mask, time=0, mtg=None):
    return FMF.f1_score(prediction, mask, average='micro', task='binary').mean().item()

@standardize_metric
def iou(prediction, mask, time=0, mtg=None):
    return FMF.jaccard_index(prediction, mask, task='binary').mean().item()

@standardize_metric
def pixel_accuracy(prediction, mask, time=0, mtg=None):
    return FMF.accuracy(prediction, mask, task='binary').mean().item()

@standardize_metric
def precision(prediction, mask, time=0, mtg=None):
    return FMF.precision(prediction, mask, task='binary').mean().item()

@standardize_metric
def recall(prediction, mask, time=0, mtg=None):
    return FMF.recall(prediction, mask, task='binary').mean().item()

@standardize_metric
def specificity(prediction, mask, time=0, mtg=None):
    # Utilise stat_scores pour obtenir tn et fp
    _, fp, tn, _, _ = FMF.stat_scores(prediction, mask, task='binary')
    spec = tn / (tn + fp + 1e-8)
    return spec.mean().item()

# =============================================================================
# Définition des métriques topology-aware
# =============================================================================

@standardize_metric
def connectivity_metric(prediction, mask, time=0, mtg=None):
    pred_np = prediction.cpu().numpy().astype(np.uint8)
    mask_np = mask.cpu().numpy().astype(np.uint8)
    conn_scores = []
    for pred_img, mask_img in zip(pred_np, mask_np):
        num_pred = label(pred_img).max()
        num_mask = label(mask_img).max()
        denominator = max(num_pred, num_mask)
        if denominator == 0:
            conn_score = 1.0 if num_pred == num_mask else 0.0
        else:
            conn_score = 1 - abs(num_pred - num_mask) / denominator
        conn_scores.append(conn_score)
    return np.mean(conn_scores).item()

@standardize_metric
def ARI_index(prediction, mask, time=0, mtg=None):
    pred_np = prediction.cpu().numpy()
    mask_np = mask.cpu().numpy()
    return adjusted_rand_score(mask_np.flatten(), pred_np.flatten())

@standardize_metric
def ARE_error(prediction, mask, time=0, mtg=None):
    pred_np = prediction.cpu().numpy()
    mask_np = mask.cpu().numpy()
    try:
        are, _, _ = adapted_rand_error(mask_np, pred_np)
    except Exception:
        are = 0.0
    return are.item()

@standardize_metric
def VI_index(prediction, mask, time=0, mtg=None):
    pred_np = prediction.cpu().numpy()
    mask_np = mask.cpu().numpy()
    H_mask = entropy(mask_np.flatten())
    H_pred = entropy(pred_np.flatten())
    MI = mutual_info_score(mask_np.flatten(), pred_np.flatten())
    VI = H_mask + H_pred - 2 * MI
    return VI

@standardize_metric
def betti_0_difference(prediction, mask, time=0, mtg=None):
    pred_np = prediction.cpu().numpy().astype(np.uint8)
    mask_np = mask.cpu().numpy().astype(np.uint8)
    scores = []
    for pred_img, mask_img in zip(pred_np, mask_np):
        num_pred = label(pred_img).max()
        num_mask = label(mask_img).max()
        scores.append(abs(num_pred - num_mask) / (num_pred + num_mask + 1e-8))
    return np.mean(scores).item()

@standardize_metric
def euler_charac_difference(prediction, mask, time=0, mtg=None):
    pred_np = prediction.cpu().numpy().astype(np.uint8)
    mask_np = mask.cpu().numpy().astype(np.uint8)
    scores = []
    for pred_img, mask_img in zip(pred_np, mask_np):
        euler_pred = euler_number(pred_img, connectivity=1)
        euler_mask = euler_number(mask_img, connectivity=1)
        scores.append(abs(euler_pred - euler_mask) / (euler_pred + euler_mask + 1e-8))
    return np.mean(scores).item()
