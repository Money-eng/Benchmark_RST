from skimage.measure import label, euler_number
from skimage.metrics import adapted_rand_error
from sklearn.metrics import adjusted_rand_score, mutual_info_score
from sklearn.metrics.cluster import entropy
import torchmetrics.functional as FMF
import torchmetrics.functional.segmentation as FMS
import numpy as np
import functools

## https://www.mdpi.com/2072-4292/16/12/2056
# APLS ? PLS is used to measure the similarity between the extracted road network and the real road network. It is defined by Equation (7). By comparing the average path lengths between them, the accuracy and completeness of the road extraction results can be evaluated, determining whether the topology of the road network is consistent with the real situation. 
# ECM ? ECM evaluates object connectivity in remote sensing road extraction by quantifying pixel relationships based on entropy. It is defined by Equation (8), where 


def all_metrics():
    return {'cpu': all_metrics_cpu(),
            'gpu': all_metrics_gpu()}


def all_metrics_gpu():
    """ 
    Returns all metrics that can be computed on GPU.
    """
    return [dice, f1_score, iou, pixel_accuracy, precision, recall, specificity]


def all_metrics_cpu():
    """ 
    Returns all metrics that can be computed on CPU.
    """
    return [connectivity_metric, ARI_index, VI_index, ARE_error,
            betti_0_difference, euler_charac_difference]

# =============================================================================
# Decorator for standardizing the inputs of metrics (conversion to int)
# =============================================================================


def standardize_metric(func):
    @functools.wraps(func)
    def wrapper(prediction, mask, time=0, mtg=None):
        # Convert to integers
        pred = prediction.int()
        msk = mask.int()
        # Remove the channel dimension if it is singular
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
            msk = msk.squeeze(1)
        return func(pred, msk, time, mtg)
    return wrapper


def standardize_float_metric(func):
    @functools.wraps(func)
    def wrapper(prediction, mask, time=0, mtg=None):
        # Convert to float
        pred = prediction.float()
        msk = mask.float()
        # Remove the channel dimension if it is singular
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
            msk = msk.squeeze(1)
        return func(pred, msk, time, mtg)
    return wrapper

# =============================================================================
# Standard Metrics Definitions
# =============================================================================


@standardize_float_metric
def dice(prediction, mask, time=0, mtg=None):
    """
    Dice Score Metric:
        Dice = (2 * |Prediction ∩ Mask|) / (|Prediction| + |Mask|)

    The function computes the Dice coefficient over the binary segmentation.
    In pseudo-math:
        dice = 2TP / (2TP + FP + FN)
    where TP, FP, FN denote true positives, false positives, and false negatives.
    """
    return FMS.dice_score(prediction, mask, num_classes=2, average='micro').mean().item()


@standardize_float_metric
def f1_score(prediction, mask, time=0, mtg=None):
    """
    F1 Score Metric:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)

    This is equivalent to the Dice coefficient in binary segmentation, and it
    combines precision and recall into one measure.
    """
    return FMF.f1_score(prediction, mask, average='micro', task='binary').mean().item()


@standardize_float_metric
def iou(prediction, mask, time=0, mtg=None):
    """
    Intersection over Union (IoU) or Jaccard Index:
        IoU = |Prediction ∩ Mask| / |Prediction ∪ Mask|

    The function calculates the Jaccard index, a common segmentation metric.
    """
    return FMF.jaccard_index(prediction, mask, task='binary').mean().item()


@standardize_float_metric
def pixel_accuracy(prediction, mask, time=0, mtg=None):
    """
    Pixel Accuracy:
        accuracy = (Number of correctly predicted pixels) / (Total number of pixels)

    This metric measures the proportion of pixels in the prediction that match the ground truth.
    """
    return FMF.accuracy(prediction, mask, task='binary').mean().item()


@standardize_float_metric
def precision(prediction, mask, time=0, mtg=None):
    """
    Precision:
        precision = TP / (TP + FP)

    Measures the fraction of predicted positive pixels that are actually positive.
    """
    return FMF.precision(prediction, mask, task='binary').mean().item()


@standardize_float_metric
def recall(prediction, mask, time=0, mtg=None):
    """
    Recall:
        recall = TP / (TP + FN)

    Measures the fraction of actual positive pixels that are correctly predicted.
    """
    return FMF.recall(prediction, mask, task='binary').mean().item()


@standardize_float_metric
def specificity(prediction, mask, time=0, mtg=None):
    """
    Specificity:
        specificity = TN / (TN + FP)

    Measures the proportion of actual negative pixels that are correctly identified.
    Here, TN and FP are computed via FMF.stat_scores.
    """
    # Use stat_scores to obtain true negatives (tn) and false positives (fp)
    _, fp, tn, _, _ = FMS.stat_scores(prediction, mask, threshold=0.5)
    spec = tn / (tn + fp + 1e-8)
    return spec.mean().item()

# =============================================================================
# Topology-Aware Metrics Definitions
# =============================================================================


@standardize_float_metric
def connectivity_metric(prediction, mask, time=0, mtg=None):
    """
    Connectivity Metric:
        Let L(P) be the label image from the prediction and L(M) from the mask.
        Let N_pred = max(label(L(P))) and N_mask = max(label(L(M))).
        The connectivity score per image is computed as:
            conn_score = 1 - |N_pred - N_mask| / max(N_pred, N_mask)
        and the final score is the mean of conn_score over the image batch.

    This metric penalizes differences in the number of connected components between the prediction and the ground truth.
    A score of 1.0 indicates perfect connectivity, while 0.0 indicates no connectivity.
    """
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


@standardize_float_metric
def ARI_index(prediction, mask, time=0, mtg=None):
    """
    Adjusted Rand Index (ARI):

    The ARI measures the similarity between two data clusterings by considering all pairs of samples and checking
    whether the clustering assignment of each pair is consistent between the predicted segmentation and the ground truth.

    It is defined as:
        ARI = (RI - Expected_RI) / (Max_RI - Expected_RI)

    where the Rand Index (RI) is given by:
        RI = (TP + TN) / (TP + TN + FP + FN)
        -> ratio entre ensembe de pixels bien segmenté sur l'ensemble de pixels
    It ranges from -1 (bad clustering) to 1 (perfect clustering).
    """
    pred_np = prediction.cpu().numpy()
    mask_np = mask.cpu().numpy()
    return adjusted_rand_score(mask_np.flatten(), pred_np.flatten())


@standardize_float_metric
def ARE_error(prediction, mask, time=0, mtg=None):
    """
    Adapted Rand Error (ARE):

    The ARE is derived from the Rand Index (RI), which computes the similarity between the predicted and ground truth segmentations
    by comparing pairwise pixel assignments. In this context, the Rand Index is defined as:
        RI = (number of pixel pairs that are similarly grouped in both prediction and ground truth) / (total number of pixel pairs)

    where the Rand Index (RI) is given by:
        RI = (TP + TN) / (TP + TN + FP + FN)

    The Adapted Rand Error is then defined as:
        ARE = 1 - RI
    """
    pred_np = prediction.cpu().numpy()
    mask_np = mask.cpu().numpy()
    try:
        are, _, _ = 1 - adapted_rand_error(mask_np, pred_np)
    except Exception:
        are = 0.0
    return are.item()


@standardize_float_metric
def VI_index(prediction, mask, time=0, mtg=None):
    """
    Variation of Information (VI):
        VI = H(Mask) + H(Prediction) - 2 * MI(Mask, Prediction)

    Where H is the entropy of the segmentation labels and MI is the mutual information.
    Lower values indicate better agreement, so we  have to negate it.
    """
    pred_np = prediction.cpu().numpy()
    mask_np = mask.cpu().numpy()
    H_mask = entropy(mask_np.flatten())
    H_pred = entropy(pred_np.flatten())
    MI = mutual_info_score(mask_np.flatten(), pred_np.flatten())
    VI = H_mask + H_pred - 2 * MI
    return 1 / (1 + VI)  # for higher score to be better


@standardize_float_metric
def betti_0_difference(prediction, mask, time=0, mtg=None):
    """
    Betti 0 Difference:
        Let N_pred = max(label(prediction)) and N_mask = max(label(mask)).
        betti_0_diff = |N_pred - N_mask| / (N_pred + N_mask + 1e-8)

    This metric measures the normalized difference in the number of connected components,
    also known as the 0th Betti number difference. A lower value means the topological structure is closer.
    """
    pred_np = prediction.cpu().numpy().astype(np.uint8)
    mask_np = mask.cpu().numpy().astype(np.uint8)
    scores = []
    for pred_img, mask_img in zip(pred_np, mask_np):
        num_pred = label(pred_img).max()
        num_mask = label(mask_img).max()
        scores.append(abs(num_pred - num_mask) / (num_pred + num_mask + 1e-8))
    mean_score = np.mean(scores).item()
    # for higer score to be better
    return 1 - mean_score


@standardize_float_metric
def euler_charac_difference(prediction, mask, time=0, mtg=None):
    """
    Euler Characteristic Difference:
        Let E_pred = euler_number(prediction) and E_mask = euler_number(mask).
        euler_diff = |E_pred - E_mask| / (E_pred + E_mask + 1e-8)

    This metric compares the Euler characteristic (which summarizes connectivity, holes, and cavities)
    between the prediction and the ground truth. Lower differences imply closer topological similarity.
    """
    pred_np = prediction.cpu().numpy().astype(np.uint8)
    mask_np = mask.cpu().numpy().astype(np.uint8)
    scores = []
    for pred_img, mask_img in zip(pred_np, mask_np):
        euler_pred = euler_number(pred_img, connectivity=1)
        euler_mask = euler_number(mask_img, connectivity=1)
        scores.append(abs(euler_pred - euler_mask) /
                      (euler_pred + euler_mask + 1e-8))
    return 1 - np.mean(scores).item()  # for higher score to be better
