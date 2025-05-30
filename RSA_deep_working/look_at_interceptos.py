import numpy as np
import os
import re

def load_all_interceptos(intercepto_dir="./interceptos"):
    """
    Charge toutes les courbes intercepto du dossier interceptos.
    Retourne un dict {(box, plant, t): np.array}
    """
    curve_dict = {}
    # Exemple de nom : box_AAA_plant_1_t_13.npy
    p = re.compile(r"box_(.+)_plant_(\d+)_t_(\d+)\.npy")
    for fname in os.listdir(intercepto_dir):
        match = p.match(fname)
        if match:
            box, plant, t = match.groups()
            arr = np.load(os.path.join(intercepto_dir, fname))
            curve_dict[(box, int(plant), int(t))] = arr
    return curve_dict

from scipy.spatial.distance import euclidean
from dtaidistance import dtw

def ccc(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * covar) / (var_true + var_pred + (mean_true - mean_pred)**2 + 1e-8)

def chamfer_1d(y1, y2):
    # Chamfer simplifiée pour séries alignées
    return np.mean(np.abs(y1 - y2))

from itertools import combinations

def all_pairwise_keys(curve_dict):
    keys = list(curve_dict.keys())
    return list(combinations(keys, 2))

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def compute_pair_metrics(args):
    k1, k2, curve_dict = args
    y1, y2 = curve_dict[k1], curve_dict[k2]
    # S'assurer que les courbes ont la même longueur
    minlen = min(len(y1), len(y2))
    y1, y2 = y1[:minlen], y2[:minlen]
    return {
        "keys": (k1, k2),
        "eucl": euclidean(y1, y2),
        "dtw": dtw.distance(y1, y2),
        "ccc": ccc(y1, y2),
        "chamfer": chamfer_1d(y1, y2),
    }

def compute_all_metrics(curve_dict, n_workers=ProcessPoolExecutor()._max_workers):
    args_list = [(k1, k2, curve_dict) for (k1, k2) in all_pairwise_keys(curve_dict)]
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for res in tqdm(executor.map(compute_pair_metrics, args_list), total=len(args_list)):
            results.append(res)
    return results

from numpy import trapz

def check_area_monotony(curve_dict):
    # Regroupe par box, plant
    results = []
    from collections import defaultdict
    plants_dict = defaultdict(dict)
    for (box, plant, t), y in curve_dict.items():
        plants_dict[(box, plant)][t] = y
    for (box, plant), times in plants_dict.items():
        times_sorted = sorted(times.items())
        prev_area = None
        for t, y in times_sorted:
            area = trapz(y)
            if prev_area is not None and area < prev_area:
                results.append({
                    "box": box,
                    "plant": plant,
                    "t": t,
                    "prev_area": prev_area,
                    "area": area,
                })
                print(f"Area violation found for box {box}, plant {plant}, t {t}: {prev_area} -> {area}")
                import matplotlib.pyplot as plt
                plt.plot(y)
                previous_y = times[t - 1]
                plt.plot(previous_y)
                plt.legend(["Current", "Previous"])
                plt.title(f"Area violation for box {box}, plant {plant}, t {t}")
                plt.show()
                
            prev_area = area
    return results

import pickle

def save_results(results, fname):
    with open(fname, "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    intercepto_dir = "./tmp_interceptos"
    curve_dict = load_all_interceptos(intercepto_dir)
    print(f"Loaded {len(curve_dict)} curves.")

    print("Checking area monotony per plant...")
    non_mono = check_area_monotony(curve_dict)
    save_results(non_mono, "monotony_violations.pkl")
    print(f"Number of area monotony violations: {len(non_mono)}")



    
