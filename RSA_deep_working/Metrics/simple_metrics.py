from skimage.measure import label
import torchmetrics.functional as FMF
import torchmetrics.functional.segmentation as FMS
import numpy as np


def all_metrics():
    return [dice, f1_score, iou, pixel_accuracy, precision, recall, specificity, connectivity_metric]

# ----------------------------
# Métriques classiques
# ----------------------------


def dice(prediction, mask, time=0, mtg=None):
    """
    Calcule le coefficient de Dice entre la prédiction et le masque.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    où A est la prédiction et B est le masque.

    Args:
        prediction (_tensor): Tenseur contenant la prédiction.
        mask (_tensor): Tenseur contenant le masque de vérité terrain.
        mtg (_type_, optional): _description_. Defaults to None.

    Returns:
        float: Le score de Dice calculé.
    """
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
    # stat_scores renvoie (TP, FP, TN, FN, tp_plus_fn)
    _, fp, tn, _, _ = FMF.stat_scores(
        prediction, mask, task='binary')
    spec = (tn.float()) / (tn.float() + fp.float())
    return spec.mean().item()


def connectivity_metric(prediction, mask, time=0, mtg=None):
    """
    Compare la connectivité en évaluant le nombre de composantes connexes.
    On calcule un score simple qui pénalise la différence entre
    le nombre de composantes connexes dans la prédiction et dans le masque.
    
    Args:
        prediction (torch.Tensor): Tenseur contenant la prédiction.
        mask (torch.Tensor): Tenseur contenant le masque de vérité terrain.
        mtg (optionnel): Valeur par défaut : None.

    Returns:
        float: Le score de connectivité calculé. (correspond à la connectivité, ie Betti score 0 ?)
    """
    prediction = prediction.int().cpu().numpy().astype(np.uint8)
    mask = mask.int().cpu().numpy().astype(np.uint8)
    conn_scores = []
    for pred_img, mask_img in zip(prediction, mask):
        label_pred = label(pred_img)
        label_mask = label(mask_img)
        n_pred = label_pred.max()  # nombre de composantes connexes
        n_mask = label_mask.max() # nombre de composantes connexes
        conn_scores.append((n_pred, n_mask))
    # somme normalisé des distances entre les composantes connexes 
    s = 0
    for i in range(len(conn_scores)):
        s += abs(conn_scores[i][0] - conn_scores[i][1])
    s /= len(conn_scores)
    return s
