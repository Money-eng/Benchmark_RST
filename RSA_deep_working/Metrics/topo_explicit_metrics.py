import torch
import numpy as np
import functools
from skimage.measure import label, euler_number
from skimage.metrics import adapted_rand_error
from sklearn.metrics import adjusted_rand_score, mutual_info_score
from sklearn.metrics.cluster import entropy
import torchmetrics.functional as FMF
import torchmetrics.functional.segmentation as FMS

###############################################################################
# Make sure to cite:
#  - clDice: : https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf
#  - Skeleton Recall Loss: : https://arxiv.org/pdf/2404.03010
#  - SuperVoxel-Based Loss for Connectivity Preservation: : https://arxiv.org/pdf/2501.01022
###############################################################################

def all_metrics():
    return [
        cldice,
        skeleton_recall,
        Connectivity_Preserving_Instance_Segmentation
    ]

###############################################################################
# Decorators to standardize input for metrics (conversion to float)
###############################################################################
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

###############################################################################
# clDice Metric Definition
###############################################################################
@standardize_float_metric
def cldice(prediction, mask, time=0, mtg=None):
    """
    clDice Metric/Loss (&#8203;:contentReference[oaicite:3]{index=3}):
    ------------------------------------------------------------------------
    - The clDice (centerline Dice) metric is specifically designed
      for measuring the connectivity of thin/tubular structures.
    - It relies on the overlap between the segmentation mask and
      its morphological skeleton.

    Pseudo-math (if used as a metric):
        1. Let P = predicted segmentation (binary after thresholding),
           and GT = ground truth.
        2. Skeletonize P --> S(P), and skeletonize GT --> S(GT).
        3. Compute Tprec = |S(P) ∩ GT| / |S(P)|
           and Tsens = |S(GT) ∩ P| / |S(GT)|.
        4. clDice = 2 * (Tprec * Tsens) / (Tprec + Tsens).

    Here, we are using an *external* soft version that can act as a loss
    term by making skeletonization differentiable. The final returned
    value can be (1 - clDice) when used as a loss.

    Implementation:
        We utilize 'soft_cldice' from an external library (imported below)
        which transforms the skeletonization process into a differentiable
        operation and computes the final soft-clDice.

    Returns:
        1 - soft_clDice(pred, mask) as a scalar float.
        Lower is better if using as a loss function.
    """
    from RSA_deep_working.Metrics.Losses.clDice.cldice_loss.pytorch.cldice import soft_cldice
    # Temporarily add a batch dimension for soft_cldice
    prediction = prediction.unsqueeze(0)
    mask = mask.unsqueeze(0)
    soft_cldice_instance = soft_cldice()
    return 1 - soft_cldice_instance(prediction, mask).item() # - to make higher is better

###############################################################################
# Skeleton Recall Metric Definition
###############################################################################
@standardize_float_metric
def skeleton_recall(prediction, mask, time=0, mtg=None):
    """
    Skeleton Recall Metric (&#8203;:contentReference[oaicite:4]{index=4}):
    ------------------------------------------------------------------------
    - The Skeleton Recall approach encourages preserving the connectivity
      of thin, tubular structures by comparing prediction vs. the ground
      truth skeleton.
    - Instead of computing a full overlap-based measure, this metric
      focuses on "how much of the GT skeleton is recalled by the prediction."

    Pseudo-math:
        SkeletonRecall =  ∑ (S(GT)_i * P_i) / ∑ S(GT)_i
        where S(GT) denotes a (possibly thickened) skeletonization
        of the ground truth mask, and P_i is the predicted mask.

    Implementation:
      - We use 'SoftSkeletonRecallLoss' which computes a differentiable
        approximation for the recall of the skeleton. The returned value
        can serve as a loss if desired.
      - For binary segmentation, the code also prepares two channels:
        (background, foreground) by concatenation.

    Returns:
        The skeleton recall value as a float. Typically, a lower result
        indicates worse matching of skeletons, so for a loss, we might
        want to maximize this recall or invert it if needed.
    """
    from RSA_deep_working.Metrics.Losses.Skeleton_Recall.nnunetv2.training.loss.dice import SoftSkeletonRecallLoss

    # We add a batch dimension for the skeleton recall loss
    prediction = prediction.unsqueeze(0)
    mask = mask.unsqueeze(0)

    soft_skeleton_recall = SoftSkeletonRecallLoss(do_bg=False)
    # Prepare 2-channel input: background = (1 - pred), foreground = pred
    prediction_2c = torch.cat([1 - prediction, prediction], dim=1)
    mask_2c = torch.cat([1 - mask, mask], dim=1)

    return soft_skeleton_recall(prediction_2c, mask_2c).item()

###############################################################################
# Connectivity Preserving Instance Segmentation Metric (SuperVoxelLoss)
###############################################################################
@standardize_float_metric
def Connectivity_Preserving_Instance_Segmentation(prediction, mask, time=0, mtg=None):
    """
    Connectivity Preserving Instance Segmentation (&#8203;:contentReference[oaicite:5]{index=5}):
    ------------------------------------------------------------------------
    - This metric (or loss) focuses on penalizing topological errors,
      especially split or merge errors, in instance segmentation tasks.
    - The approach extends the concept of non-simple voxels to "supervoxels":
      connected sets of voxels whose addition/removal changes the total
      number of connected components.

    Pseudo-math (sketch):
        1. Identify the false-positive (FP) and false-negative (FN) regions.
        2. Within these FP/FN volumes, find the connected components
           that cause connectivity changes (called "critical components").
        3. Impose additional penalties (via L0) on each critical component
           to preserve correct connectivity.

    Implementation:
        The code below uses 'SuperVoxelLoss2D' with user-defined parameters
        alpha=0.5 and beta=0.5, to demonstrate a basic weighting between
        voxel-level and topological mistakes. It is applied to 2D data.
    
    Returns:
        The scalar loss value as a float, indicating the topological penalty
        for connectivity preservation. Lower is better.
    """
    from RSA_deep_working.Metrics.Losses.supervoxel_loss.src.supervoxel_loss.loss import SuperVoxelLoss2D

    # Instantiate the Supervoxel loss
    SuperVoxelLoss = SuperVoxelLoss2D(alpha=0.5, beta=0.5)

    # Add a batch dimension for the loss
    prediction = prediction.unsqueeze(0)
    mask = mask.unsqueeze(0)

    # Compute the supervoxel-based connectivity-preserving loss
    return - SuperVoxelLoss(prediction, mask).item() # - to make higher is better


@standardize_float_metric
def evaluate_sk_seg(prediction, mask, time=0, mtg_path=None):
    # Exemple d'évaluation sur la segmentation "skeletonisée"
    import numpy as np
    from rsml import rsml2mtg
    mtg = rsml2mtg(mtg_path)
    prediction = prediction.int().cpu().numpy().astype(np.uint8)
    mask = mask.int().cpu().numpy().astype(np.uint8)
    prediction = np.squeeze(prediction)
    mask = np.squeeze(mask)

    def vertex_cc_2_mtg(mtg, mask, time0):
        """ 
        Extrait les composantes connexes dans le MTG associé au masque à un certain temps.
            S'il y a un noeud d'une racine qui se trouve sur une composante connexe, on associe à la racine le label de la composante connexe dans un dictionnaire.
        """
        from skimage.measure import label
        labeled_mask = label(mask) # renvoi une image avec des labels uniques pour chaque composante connexe
        roots_2_cc = {}
        for vertex in mtg.vertices(): # pour chaque racines du MTG (vertex scale 1)
            try: # on récupère la géométrie et le temps des racines
                geometry = mtg[vertex].get("geometry")
                times = mtg[vertex].get("time")
            except KeyError:
                continue
            if geometry is None:
                continue
            cc_2_root = set()
            for pos in geometry: # pour chaque noeud de la racine
                x, y = pos
                time = times[geometry.index(pos)]
                try:
                    if time > time0: # si le temps est bien défini
                        continue
                    connected_component = labeled_mask[int(y), int(x)] # on récupère le label de la composante connexe associée à ce pixel
                    cc_2_root.add(connected_component)
                except IndexError:
                    continue
            roots_2_cc[vertex] = cc_2_root

        return roots_2_cc # un dictionnaire avec les racines du MTG et les labels des composantes connexes associées
    
    if prediction.ndim == 2:
        cc_gt = vertex_cc_2_mtg(mtg, mask, time) # on récupère les composantes connexes graph / ground truth
        cc_pred = vertex_cc_2_mtg(mtg, prediction, time) # on récupère les composantes connexes graph / prédiction
        # et on compare les deux (décompte des "faux positifs" et "faux négatifs")
        count_bad_conn = 0
        count_good_conn = 0
        for root_cc in cc_gt:
            if root_cc not in cc_pred: # une composante connexe absente de la prédiction totale
                count_bad_conn += 1
            elif cc_gt[root_cc] != cc_pred[root_cc]: # pour la racine cc, les composantes connexes ne sont pas les mêmes
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

