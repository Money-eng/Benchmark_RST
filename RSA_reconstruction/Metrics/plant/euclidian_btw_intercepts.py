# Metrics/cpu/dtw_between_intercepts.py
import numpy as np
from openalea.mtg import MTG


# from utils.intercept import intercept_curve_at_all_time
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


class EuclidianDistancebtwIntercepts():
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Dynamic Time Warping (DTW) between intercepts. On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return new_score < old_score

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        import numpy as np
        from scipy.spatial.distance import euclidean

        plant_scale = 1
        verts_gt = list(mtg_gt.vertices(scale=plant_scale))
        verts_pred = list(mtg_pred.vertices(scale=plant_scale))

        # Construire les sous-MTG pour chaque racine
        map_subtree_gt = {v: mtg_gt.sub_mtg(v) for v in verts_gt}
        map_subtree_pred = {v: mtg_pred.sub_mtg(v) for v in verts_pred}

        # Calcul des courbes d’interception pour chaque sous-arbre
        map_curve_gt = {v: intercept_curve_at_all_time(map_subtree_gt[v], 0)
                        for v in verts_gt}
        map_curve_pred = {v: intercept_curve_at_all_time(map_subtree_pred[v], 0)
                          for v in verts_pred}

        distances = []
        all_vs = set(map_curve_gt.keys()) | set(map_curve_pred.keys())
        for v in all_vs:
            # Récupérer (x, y) ; si absent, y_pred ou y_gt devient matrice nulle de même forme
            if v in map_curve_gt:
                x_gt, y_gt = map_curve_gt[v]
            else:
                x_gt, y_gt = map_curve_pred[v]
                y_gt = np.zeros_like(y_gt)

            if v in map_curve_pred:
                x_pred, y_pred = map_curve_pred[v]
            else:
                x_pred, y_pred = map_curve_gt[v]
                y_pred = np.zeros_like(y_pred)

            # Si les grilles x diffèrent, on ré-échantillonne y_pred sur x_gt
            if not np.array_equal(x_gt, x_pred):
                # interpolation ligne à ligne (sur l’axe des distances)
                y_pred_interp = np.vstack([
                    np.interp(x_gt, x_pred, y_pred_t)
                    for y_pred_t in y_pred
                ])
            else:
                y_pred_interp = y_pred

            # Aligner la dimension temps (prendre le plus petit)
            T_common = min(y_gt.shape[0], y_pred_interp.shape[0])
            y1 = y_gt[:T_common].flatten()
            y2 = y_pred_interp[:T_common].flatten()

            distances.append(euclidean(y1, y2))

        # Retourner la distance moyenne sur tous les sous-arbres
        return float(np.mean(distances))


if __name__ == "__main__":
    from rsml import rsml2mtg

    mtg_gt = rsml2mtg("/home/loai/Images/DataTest/UC1_data/Train/230629PN011/61_graph.rsml")
    mtg_pred = rsml2mtg("/home/loai/Images/DataTest/UC1_data/Train/230629PN011/61_graph.rsml")
    # remove 1 index from mtg_pred (max scale)
    mtg_pred.remove_vertex(mtg_pred.vertices(scale=mtg_pred.max_scale())[-1])  # remove

    metric = EuclidianDistancebtwIntercepts()
    score = metric(mtg_pred, mtg_gt)
    print(f"L2-distance between intercepts: {score}")
