# ----------------------------
# Métriques additionnelles pour la segmentation tubulaire
# ----------------------------
import torch
import functools

def all_metrics():
    return [cldice, skeleton_recall, Connectivity_Preserving_Instance_Segmentation, evaluate_sk_seg]


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
# Définition des métriques tubulaires (topology-aware spécifiques)
# =============================================================================

@standardize_float_metric
def cldice(prediction, mask, time=0, mtg=None):
    # Exemple avec une implémentation externe (assurez-vous que le module est installé)
    from RSA_deep_working.Metrics.Losses.clDice.cldice_loss.pytorch.cldice import soft_cldice
    prediction = prediction.unsqueeze(0)
    mask = mask.unsqueeze(0)
    soft_cldice_instance = soft_cldice()
    return 1 - soft_cldice_instance(prediction, mask).item()

@standardize_float_metric
def skeleton_recall(prediction, mask, time=0, mtg=None):
    from RSA_deep_working.Metrics.Losses.Skeleton_Recall.nnunetv2.training.loss.dice import SoftSkeletonRecallLoss
    prediction = prediction.unsqueeze(0)
    mask = mask.unsqueeze(0)
    soft_skeleton_recall = SoftSkeletonRecallLoss(do_bg=False)
    # Préparation pour 2 canaux : background et prédiction
    prediction_2c = torch.cat([1 - prediction, prediction], dim=1)
    mask_2c = torch.cat([1 - mask, mask], dim=1)
    return soft_skeleton_recall(prediction_2c, mask_2c).item()

@standardize_float_metric
def Connectivity_Preserving_Instance_Segmentation(prediction, mask, time=0, mtg=None):
    from RSA_deep_working.Metrics.Losses.supervoxel_loss.src.supervoxel_loss.loss import SuperVoxelLoss2D
    SuperVoxelLoss = SuperVoxelLoss2D()
    prediction = prediction.unsqueeze(0)
    mask = mask.unsqueeze(0)
    return SuperVoxelLoss(prediction, mask).item()

@standardize_float_metric
def evaluate_sk_seg(prediction, mask, time=0, mtg_path=None):
    # Exemple d'évaluation sur la segmentation skeletonisée
    import numpy as np
    from rsml import rsml2mtg
    mtg = rsml2mtg(mtg_path)
    prediction = prediction.int().cpu().numpy().astype(np.uint8)
    mask = mask.int().cpu().numpy().astype(np.uint8)
    prediction = np.squeeze(prediction)
    mask = np.squeeze(mask)
    
    # Définition locale d'une fonction pour extraire les composantes connexes
    def vertex_cc_2_mtg(mtg, mask, time0):
        from skimage.measure import label
        labeled_mask = label(mask)
        roots_2_cc = {}
        for vertex in mtg.vertices():
            try:
                geometry = mtg[vertex].get("geometry")
                times = mtg[vertex].get("time")
            except KeyError:
                continue
            if geometry is None:
                continue
            cc_2_root = set()
            for pos in geometry:
                x, y = pos
                time = times[geometry.index(pos)]
                try:
                    if time > time0:
                        continue
                    connected_component = labeled_mask[int(y), int(x)]
                    cc_2_root.add(connected_component)
                except IndexError:
                    continue
            roots_2_cc[vertex] = cc_2_root
        return roots_2_cc
    
    if prediction.ndim == 2:
        cc_gt = vertex_cc_2_mtg(mtg, mask, time)
        cc_pred = vertex_cc_2_mtg(mtg, prediction, time)
        count_bad_conn = 0
        count_good_conn = 0
        for cc in cc_gt:
            if cc not in cc_pred:
                count_bad_conn += 1
            elif cc_gt[cc] != cc_pred[cc]:
                count_bad_conn += 1
            else:
                count_good_conn += 1
        return count_good_conn / (count_bad_conn + count_good_conn + 1e-8)
    else:
        total_good = 0
        total_bad = 0
        for i in range(prediction.shape[0]):
            cc_gt = vertex_cc_2_mtg(mtg, mask[i], time[i])
            cc_pred = vertex_cc_2_mtg(mtg, prediction[i], time[i])
            for cc in cc_gt:
                if cc not in cc_pred:
                    total_bad += 1
                elif cc_gt[cc] != cc_pred[cc]:
                    total_bad += 1
                else:
                    total_good += 1
        return total_good / (total_good + total_bad + 1e-8)