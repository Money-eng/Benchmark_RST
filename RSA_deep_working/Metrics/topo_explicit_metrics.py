# ----------------------------
# Métriques additionnelles pour la segmentation tubulaire
# ----------------------------
import torch

def all_metrics():
    return [cldice, skeleton_recall, Connectivity_Preserving_Instance_Segmentation, evaluate_sk_seg]


def cldice(prediction, mask, time=0, mtg=None):
    """
    Calcule la métrique clDice, une variante du coefficient Dice spécialement conçue pour
    évaluer la segmentation de structures tubulaires. Cette métrique intègre à la fois la 
    connectivité et la forme, deux aspects essentiels pour une évaluation précise en imagerie médicale.

    Le score clDice est défini par la formule suivante :
        clDice = (2 * TP(Sp, Vl) * TS(Sl, Vp)) / (TP(Sp, Vl) + TS(Sl, Vp))
    où :
        - TP (Précision Topologique) correspond à |A ∩ B| / |A|, avec A représentant le squelette prédit
          et B la vérité terrain.
        - TS (Sensibilité Topologique) correspond à |A ∩ B| / |A|, avec A représentant le squelette extrait 
          de la vérité terrain et B la prédiction.

    Args:
        prediction (torch.Tensor): Tenseur contenant la sortie de la segmentation prédite.
        mask (torch.Tensor): Tenseur contenant la segmentation de la vérité terrain.
        mtg (optionnel): Structure additionnelle (par exemple, un multigraph) requise pour certains calculs.
                       Valeur par défaut : None.

    Returns:
        float: Le score clDice calculé.
    """
    from RSA_deep_working.Metrics.Losses.clDice.cldice_loss.pytorch.cldice import soft_cldice
    prediction = prediction.unsqueeze(0)
    mask = mask.unsqueeze(0)
    soft_cldice = soft_cldice()
    return 1 - soft_cldice(prediction, mask).item()


def skeleton_recall(prediction, mask, time=0, mtg=None):
    """
    Calcule la métrique de rappel du squelette, qui mesure la capacité de la prédiction à 
    recouvrir le squelette (ou structure tubulaire) extrait de la vérité terrain.

    La loss de rappel du squelette est formulée comme suit :
        L = - (1 / |C|) * ∑ (|A ∩ B| / |A|)
    où :
        - A représente le squelette tubulaire extrait de la vérité terrain.
        - B représente la prédiction du réseau.
        - |C| correspond au nombre de classes, assurant une normalisation sur l'ensemble des classes.

    Args:
        prediction (torch.Tensor): Tenseur contenant la segmentation prédite.
        mask (torch.Tensor): Tenseur contenant la segmentation de la vérité terrain.
        mtg (optionnel): Structure additionnelle éventuellement requise pour le calcul.
                       Valeur par défaut : None.

    Returns:
        float: La valeur de la loss de rappel du squelette calculée.
    """
    from RSA_deep_working.Metrics.Losses.SkeletonRecall.nnunetv2.training.loss.dice import SoftSkeletonRecallLoss
    
    prediction = prediction.unsqueeze(0)
    mask = mask.unsqueeze(0)
    soft_skeleton_recall = SoftSkeletonRecallLoss(do_bg=False)
    # Pour la prédiction & mask : le premier canal (background) est 1 - prédiction, le second canal est la prédiction
    prediction_2c = torch.cat([1 - prediction, prediction], dim=1)
    mask_2c = torch.cat([1 - mask, mask], dim=1)
    return soft_skeleton_recall(prediction_2c, mask_2c).item()


def Connectivity_Preserving_Instance_Segmentation(prediction, mask, time=0, mtg=None):
    from RSA_deep_working.Metrics.Losses.supervoxel_loss.src.supervoxel_loss.loss import SuperVoxelLoss2D
    SuperVoxelLoss = SuperVoxelLoss2D()
    prediction = prediction.unsqueeze(0)
    mask = mask.unsqueeze(0)
    return SuperVoxelLoss(prediction, mask).item()


# source des idées https://github.com/AllenNeuralDynamics/segmentation-skeleton-metrics/
def evaluate_sk_seg(prediction, mask, time=0, mtg_path=None):
    import numpy as np
    from rsml import rsml2mtg
    mtg = rsml2mtg(mtg_path)
    prediction = prediction.int().cpu().numpy().astype(np.uint8)
    mask = mask.int().cpu().numpy().astype(np.uint8)
    mask = np.squeeze(mask)
    prediction = np.squeeze(prediction)
    cc_gt = vertex_cc_2_mtg(mtg, mask, time)
    cc_pred = vertex_cc_2_mtg(mtg, prediction, time)
    # count the number of not predicted connected components
    count_bad_conn = 0
    cound_good_conn = 0
    for cc in cc_gt:
        if cc not in cc_pred:
            count_bad_conn += 1
        elif cc_gt[cc] != cc_pred[cc]:
            count_bad_conn += 1
        else:
            cound_good_conn += 1
    # return ratio of good connected components over the total number of connected components
    return cound_good_conn / (count_bad_conn + cound_good_conn)


def vertex_cc_2_mtg(mtg, mask, time0):
    # for each key in the hierarchy, get the connected components of the mask
    from skimage.measure import label
    import numpy as np
    labeled_mask = label(mask)
    roots_2_cc = {}
    for vertex in mtg.vertices():
        try:
            geometry = mtg[vertex].get("geometry")
            times = mtg[vertex].get("time")
        except KeyError:
            print("KeyError: geometry not found")
            continue

        if geometry is None:
            # print("Geometry is None")
            continue
        cc_2_root = set()
        count_0 = 0
        count_label = 0
        # for all the positions of the geometry, get the corresponding connected components of the mask
        for pos in geometry:
            x, y = pos
            time = times[geometry.index(pos)]
            # get the connected component of the mask
            try:
                if time > time0:
                    continue
                connected_component = labeled_mask[int(y), int(x)]
                cc_2_root.add(connected_component)
                if connected_component == 0:
                    count_0 += 1
                else:
                    count_label += 1
            except IndexError:
                print("IndexError: position out of bounds :",
                      pos, "for image size", mask.shape)
                continue
        roots_2_cc[vertex] = cc_2_root
    # same principle, but we trace the poliline of the root geometry and we select all the cc the polyline intersects
    for vertex in mtg.vertices():
        try:
            geometry = mtg[vertex].get("geometry")
            times = mtg[vertex].get("time")
        except KeyError:
            print("KeyError: geometry not found")
            continue
        if geometry is None:
            # print("Geometry is None")
            continue
        cc_2_root = set()
        count_0 = 0
        count_label = 0
        # for all the positions of the geometry, get the corresponding connected components of the mask
        for index_pos in range(len(geometry)-1):
            x1, y1 = geometry[index_pos]
            x2, y2 = geometry[index_pos+1]
            time = times[geometry.index(geometry[index_pos+1])]
            # get the connected component of the mask
            try:
                if time > time0:
                    continue
                # get the connected that intersects the line between (x1, y1) and (x2, y2)
                # create a line between the two points
                x_line = np.linspace(x1, x2, num=100).astype(int)
                y_line = np.linspace(y1, y2, num=100).astype(int)
                # get the connected component of the mask
                labeled_mask_line = labeled_mask[y_line, x_line]
                # get the unique connected components of the mask
                connected_component = np.unique(labeled_mask_line)
                cc_2_root.update(connected_component)
                if 0 in connected_component:
                    count_0 += 1
                else:
                    count_label += 1
            except IndexError:
                print("IndexError: position out of bounds")
                continue
    return roots_2_cc
