# Metrics/cpu/ari_index.py
import numpy as np
from openalea.mtg import MTG
# from utils.intercept import intercept_curve_at_all_time
from scipy.optimize import linear_sum_assignment

from ..base import BaseMetric


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


class AreaBetweenIntercepts(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """Plus petit est meilleur."""
        return new_score < old_score

    @staticmethod
    def _area_between_surfaces(x1: np.ndarray, Y1: np.ndarray,
                               x2: np.ndarray, Y2: np.ndarray) -> float:
        """
        Calcule l’aire de |Y1(t,x1) - Y2(t,x2)| sur (t,x) par double trapz.
        Y arrays shape = (n_times, n_lengths).
        """
        # si besoin, interpole Y2 sur la grille x1
        if not np.array_equal(x1, x2):
            Y2 = np.vstack([np.interp(x1, x2, row) for row in Y2])
        # vecteur temps équidistant de 1, 2, …, n_times
        times = np.arange(1, Y1.shape[0] + 1)
        # trapz sur x puis trapz sur t
        A_x = np.trapz(np.abs(Y1 - Y2), x=x1, axis=1)
        return float(np.trapz(A_x, x=times))

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        plant_scale = 1
        verts_gt = list(mtg_gt.vertices(scale=plant_scale))
        verts_pred = list(mtg_pred.vertices(scale=plant_scale))

        # préparer les sous-arbres
        sub_gt = {v: mtg_gt.sub_mtg(v) for v in verts_gt}
        sub_pred = {v: mtg_pred.sub_mtg(v) for v in verts_pred}

        # calculer toutes les courbes (x, Y) pour chaque sous-arbre
        curves_gt = {v: intercept_curve_at_all_time(sub_gt[v], 0)
                     for v in verts_gt}
        curves_pred = {v: intercept_curve_at_all_time(sub_pred[v], 0)
                       for v in verts_pred}

        n_gt, n_pr = len(verts_gt), len(verts_pred)
        cost = np.zeros((n_gt, n_pr), dtype=float)

        # remplir la matrice de coût
        for i, v_gt in enumerate(verts_gt):
            x_gt, Y_gt = curves_gt[v_gt]
            for j, v_pr in enumerate(verts_pred):
                x_pr, Y_pr = curves_pred[v_pr]
                cost[i, j] = self._area_between_surfaces(x_gt, Y_gt, x_pr, Y_pr)

        # appariement optimal
        row_ind, col_ind = linear_sum_assignment(cost)

        total = cost[row_ind, col_ind].sum()

        # gérer les GT non appariés (si n_gt > n_pr)
        if n_gt > n_pr:
            unmatched = set(range(n_gt)) - set(row_ind)
            for i in unmatched:
                x_gt, Y_gt = curves_gt[verts_gt[i]]
                # aire entre GT et surface nulle
                total += self._area_between_surfaces(x_gt, Y_gt, x_gt, np.zeros_like(Y_gt))

        return float(total)


# test
if __name__ == "__main__":
    from rsml import rsml2mtg

    mtg_gt = rsml2mtg("/home/loai/Images/DataTest/UC1_data/Train/230629PN011/61_graph.rsml")
    mtg_pred = rsml2mtg("/home/loai/Images/DataTest/UC1_data/Train/230629PN011/61_graph.rsml")
    # remove 1 index from mtg_pred (max scale)
    mtg_pred.remove_vertex(mtg_pred.vertices(scale=mtg_pred.max_scale())[-1])  # remove

    metric = AreaBetweenIntercepts()
    score = metric(mtg_pred, mtg_gt)
    print(f"Area between intercepts: {score}")
