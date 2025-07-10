# Metrics/cpu/dtw_between_intercepts.py
import numpy as np
# from utils.intercept import intercept_curve_at_all_time
from fastdtw import fastdtw
from openalea.mtg import MTG
from scipy.optimize import linear_sum_assignment


# from ..base import BaseMetric


def mtg_at_time_t(mtg: MTG, temps_max: float) -> MTG:
    """
    Create a new MTG with only the vertices that are present at a given time.
    """
    new_g = mtg.copy()
    to_remove = []

    for v in new_g.vertices(new_g.max_scale()):  # scale=2, normalement, c’est chaque axe/racine
        node = new_g.node(v)
        if hasattr(node, "time"):
            t = node.time
        else:
            # DEBATABLE
            continue

        first_t = min(t)
        if first_t > temps_max:
            to_remove.append(v)
            continue

        mask = [tt <= temps_max for tt in t]
        if hasattr(node, "geometry"):
            node.geometry = [p for p, m in zip(node.geometry, mask) if m]
        if hasattr(node, "diameter"):
            node.diameter = [d for d, m in zip(node.diameter, mask) if m]
        node.time = [tt for tt, m in zip(t, mask) if m]
        node.time_hours = [th for th, m in zip(node.time_hours, mask) if m]

        if not node.geometry or len(node.geometry) < 2:
            to_remove.append(v)

    # On enlève toutes les racines/axes à supprimer
    for v in to_remove:
        new_g.remove_vertex(v)  # ou new_g.delete_vertex(v) selon la lib

    return new_g


def intercept_curve_at_all_time(mtg: MTG, plant_id=1, nlengths=2500, step=1e-3):
    """
    Calcule la courbe intercepto pour une plante d'un mtg, éventuellement à un temps donné.
    """
    from hydroroot.analysis import intercept
    from hydroroot.hydro_io import import_rsml_to_discrete_mtg
    times = mtg.properties()["time"]
    # get max time value from dict
    max_time = max(max(times.values()))
    times = [i for i in range(1, int(max_time) + 1)]
    lengths = np.linspace(0, (nlengths - 1) * step, nlengths)
    intercepto_all = []
    for time in times:
        sub_mtg = mtg.sub_mtg(plant_id)
        mtg_at_t = mtg_at_time_t(sub_mtg, time)
        mtg_test = import_rsml_to_discrete_mtg(mtg_at_t)
        intercepto = intercept(g=mtg_test, dists=lengths,
                               dl=3e-3, max_order=None)
        intercepto_all.append(intercepto)
    intercepto_all = np.array(intercepto_all)
    return lengths, intercepto_all


class DTWBetweenIntercepts():
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """Plus le score est petit, meilleur c'est."""
        return new_score < old_score

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        # 1) Récupération des racines (scale=1)
        scale = 1
        verts_gt = list(mtg_gt.vertices(scale=scale))
        verts_pred = list(mtg_pred.vertices(scale=scale))

        # 2) Extraction des sous-arbres
        sub_gt = {v: mtg_gt.sub_mtg(v) for v in verts_gt}
        sub_pred = {v: mtg_pred.sub_mtg(v) for v in verts_pred}

        # 3) Calcul des courbes (x, y) pour chaque sous-arbre
        curves_gt = {}
        for v, g in sub_gt.items():
            x, y = intercept_curve_at_all_time(g, 0)
            curves_gt[v] = (x, y)

        curves_pred = {}
        for v, g in sub_pred.items():
            x, y = intercept_curve_at_all_time(g, 0)
            curves_pred[v] = (x, y)

        # 4) Construction de la matrice de coûts
        n_gt, n_pred = len(verts_gt), len(verts_pred)
        cost = np.zeros((n_gt, n_pred), dtype=float)

        for i, v_gt in enumerate(verts_gt):
            x_gt, y_gt = curves_gt[v_gt]  # y_gt : (T_gt, L)
            for j, v_pred in enumerate(verts_pred):
                x_pr, y_pr = curves_pred[v_pred]  # y_pr : (T_pr, L')

                # 4a) Rééchantillonnage sur x_gt si nécessaire
                if not np.array_equal(x_gt, x_pr):
                    # pour chaque pas de temps, interpole sur la grille x_gt
                    y_pr = np.vstack([
                        np.interp(x_gt, x_pr, row)
                        for row in y_pr
                    ])

                # 4b) Séquences de vecteurs 
                seq_gt = [tuple(row) for row in y_gt]
                seq_pr = [tuple(row) for row in y_pr]

                # 4c) DTW sur la séquence de vecteurs
                dist, _ = fastdtw(seq_gt, seq_pr)  # retourne la distance DTW (un float)
                cost[i, j] = dist

        row_ind, col_ind = linear_sum_assignment(
            cost)  # retourne les indices optimaux - ceux qui minimisent la somme des coûts
        if row_ind != col_ind:
            raise ValueError("Indices de lignes et de colonnes ne correspondent pas.")
        print(f"DTW cost matrix:\n{cost}")
        print(f"Row indices: {row_ind}")
        print(f"Column indices: {col_ind}")
        total = cost[row_ind, col_ind].sum()
        print(f"Total DTW cost: {total} avec {cost[row_ind, col_ind]}")
        return total


if __name__ == "__main__":
    from rsml import rsml2mtg

    mtg_gt = rsml2mtg("/home/loai/Images/DataTest/UC1_data/Train/230629PN011/61_graph.rsml")
    mtg_pred = rsml2mtg("/home/loai/Images/DataTest/UC1_data/Train/230629PN011/61_graph.rsml")
    # remove 1 index from mtg_pred (max scale)
    mtg_pred.remove_vertex(mtg_pred.vertices(scale=mtg_pred.max_scale())[-1])  # remove

    metric = DTWBetweenIntercepts()
    score = metric(mtg_pred, mtg_gt)
    print(f"DTW between intercepts: {score}")
