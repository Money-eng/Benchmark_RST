import torch
import torchmetrics.functional as FMF
import torchmetrics.functional.segmentation as FMS
import numpy as np


def all_metrics():
    return [dice, f1_score, iou, pixel_accuracy, precision, recall, specificity, connectivity_metric]

# ----------------------------
# Métriques classiques
# ----------------------------


def dice(prediction, mask, mtg=None):
    prediction = prediction.int()
    mask = mask.int()
    return FMS.dice_score(prediction, mask, num_classes=2, average='micro').mean().item()


def f1_score(prediction, mask, mtg=None):
    prediction = prediction.int()
    mask = mask.int()
    return FMF.f1_score(prediction, mask, average='micro', task='binary').mean().item()

def iou(prediction, mask, mtg=None):
    prediction = prediction.int()
    mask = mask.int()
    return FMF.jaccard_index(prediction, mask, task='binary').mean().item()


def pixel_accuracy(prediction, mask, mtg=None):
    prediction = prediction.int()
    mask = mask.int()
    return FMF.accuracy(prediction, mask, task='binary').mean().item()


def precision(prediction, mask, mtg=None):
    prediction = prediction.int()
    mask = mask.int()
    return FMF.precision(prediction, mask, task='binary').mean().item()


def recall(prediction, mask, mtg=None):
    prediction = prediction.int()
    mask = mask.int()
    return FMF.recall(prediction, mask, task='binary').mean().item()


def specificity(prediction, mask, mtg=None):
    prediction = prediction.int()
    mask = mask.int()
    # stat_scores renvoie (TP, FP, TN, FN, tp_plus_fn)
    tp, fp, tn, fn, tp_plus_fn = FMF.stat_scores(
        prediction, mask, task='binary')
    spec = (tn.float()) / (tn.float() + fp.float())
    return spec.mean().item()

from skimage.measure import label
def connectivity_metric(prediction, mask, mtg=None):
    """
    Compare la connectivité en évaluant le nombre de composantes connexes.
    On calcule un score simple qui pénalise la différence entre
    le nombre de composantes connexes dans la prédiction et dans le masque.
    """
    prediction = prediction.int().cpu().numpy().astype(np.uint8)
    mask = mask.int().cpu().numpy().astype(np.uint8)
    conn_scores = []
    for pred_img, mask_img in zip(prediction, mask):
        label_pred = label(pred_img)
        label_mask = label(mask_img)
        n_pred = label_pred.max()  # nombre de composantes
        n_mask = label_mask.max()
        if n_mask == 0:
            score = 1.0 if n_pred == 0 else 0.0
        else:
            # Plus la différence est faible, plus le score est proche de 1.
            score = abs(1.0 - abs(n_pred - n_mask) / n_mask)
        conn_scores.append(score)
    return np.mean(conn_scores)
