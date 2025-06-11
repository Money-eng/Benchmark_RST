import numpy as np
from pathlib import Path


def load_mtg_from_rsml(rsml_path: str):
    """
    Charge un MTG depuis un fichier RSML.
    """
    from rsml import rsml2mtg
    return rsml2mtg(rsml_path)


def compute_interceptos_for_mtg(mtg, times=range(1, 29), nlengths=2500):
    """
    Pour chaque plante et chaque temps, calcule la courbe intercepto.
    Retourne: dict {plant_id: {t: intercepto_curve}}
    """
    from hydroroot.analysis import intercept
    from hydroroot.hydro_io import import_rsml_to_discrete_mtg
    from util.mtgutils import mtg_at_time_t

    res = {}
    plant_ids = mtg.vertices(scale=1)
    list_lengths = np.linspace(0, (nlengths - 1) * 1e-3, nlengths)
    for plant_id in plant_ids:
        res[plant_id] = {}
        sub_mtg = mtg.sub_mtg(plant_id)
        for t in times:
            mtg_at_t = mtg_at_time_t(sub_mtg, t)
            mtg_discrete = import_rsml_to_discrete_mtg(mtg_at_t)
            intercepto = intercept(g=mtg_discrete, dists=list_lengths, dl=3e-3, max_order=None)
            res[plant_id][t] = np.array(intercepto)
    return res, list_lengths


# ========================
# METRICS FUNCTIONS
# ========================
from scipy.spatial.distance import euclidean
from dtaidistance import dtw


def ccc(y1, y2):
    y1, y2 = np.array(y1), np.array(y2)
    m1, m2 = y1.mean(), y2.mean()
    v1, v2 = y1.var(), y2.var()
    cov = ((y1 - m1) * (y2 - m2)).mean()
    return 2 * cov / (v1 + v2 + (m1 - m2) ** 2 + 1e-8)


def chamfer_1d(y1, y2):
    return np.mean(np.abs(y1 - y2))


def l1_area(y1, y2, x):
    return np.trapz(np.abs(y1 - y2), x)


def l2_area(y1, y2, x):
    return np.sqrt(np.trapz((y1 - y2) ** 2, x))


# ========================
# METRIC COMPUTATION
# ========================

def compute_all_metrics_for_plant(intercepto_dict, list_lengths):
    """
    Pour une plante donnée (dict t:curve), calcule toutes les métriques pour toutes les paires de temps.
    Retourne une liste de dicts.
    """
    results = []
    times = sorted(intercepto_dict.keys())
    for i, t1 in enumerate(times):
        y1 = intercepto_dict[t1]
        for t2 in times[i + 1:]:
            y2 = intercepto_dict[t2]
            minlen = min(len(y1), len(y2))
            y1c, y2c = y1[:minlen], y2[:minlen]
            x = list_lengths[:minlen]
            res = {
                "t1": t1,
                "t2": t2,
                "euclidean": euclidean(y1c, y2c),
                "dtw": dtw.distance(y1c, y2c),
                "ccc": ccc(y1c, y2c),
                "chamfer": chamfer_1d(y1c, y2c),
                "l1_area": l1_area(y1c, y2c, x),
                "l2_area": l2_area(y1c, y2c, x),
                "area_t1": np.trapz(y1c, x),
                "area_t2": np.trapz(y2c, x),
                "monotonic": np.trapz(y2c, x) >= np.trapz(y1c, x)
            }
            results.append(res)
            print(f"Comparing t1={t1} and t2={t2}: {res}")
    return results


# ========================
# MAIN FUNCTION
# ========================

def main(rsml_path):
    print(f"Loading {rsml_path}")
    mtg = load_mtg_from_rsml(rsml_path)
    interceptos, list_lengths = compute_interceptos_for_mtg(mtg)

    all_results = {}
    for plant_id, t_dict in interceptos.items():
        metrics = compute_all_metrics_for_plant(t_dict, list_lengths)
        all_results[plant_id] = metrics

        # Display violations of monotonicity
        for res in metrics:
            if not res['monotonic']:
                print(f"Monotonicity violated for plant {plant_id} between t={res['t1']} and t={res['t2']} "
                      f"(area_t1={res['area_t1']:.2f}, area_t2={res['area_t2']:.2f})")

    # Optionally save results as npz or pickle
    # np.savez("metrics_per_plant.npz", **all_results)
    # with open("metrics_per_plant.pkl", "wb") as f:
    #     import pickle; pickle.dump(all_results, f)

    print("Done. Example result (first plant):")
    first_plant = list(all_results.keys())[0]
    print(all_results[first_plant][:3])  # Affiche les 3 premières comparaisons


if __name__ == "__main__":
    rsml_path = "/home/loai/Images/DataTest/UC1_data_backup/230629PN024/61_graph.rsml"
    main(rsml_path)
