from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from hydroroot.analysis import intercept

def find_rsml_paths(base_directory):
    return [str(p) for p in Path(base_directory).rglob("*.rsml")]

def compute_intercepto_for_file(mtg_path: str, times=None, nlengths=2500):
    from rsml import rsml2mtg
    from hydroroot.hydro_io import import_rsml_to_discrete_mtg
    from util.mtgutils import mtg_at_time_t

    if times is None:
        times = [i for i in range(1, 29)]
    mtg_box_name = mtg_path.split("/")[-2]
    mtg = rsml2mtg(mtg_path)
    plant_ids = mtg.vertices(scale=1)
    res = {}
    for plant_id in plant_ids:
        res[plant_id] = {}
        sub_mtg_contninous = mtg.sub_mtg(plant_id)
        for time in times:
            time_mtg0 = mtg_at_time_t(sub_mtg_contninous, time)
            time_mtg = import_rsml_to_discrete_mtg(time_mtg0)
            list_lengths = [i * 1e-3 for i in range(nlengths)]
            intercepto = intercept(g=time_mtg, dists=list_lengths, dl=3e-3, max_order=None)
            res[plant_id][time] = intercepto
    return mtg_box_name, res

def save_interceptos(intercepto_all, temp_dir="./tmp_interceptos"):
    import numpy as np
    import os
    os.makedirs(temp_dir, exist_ok=True)
    for box, plants in intercepto_all.items():
        for plant, times in plants.items():
            for t, y in times.items():
                filename = f"{temp_dir}/box_{box}_plant_{plant}_t_{t}.npy"
                np.save(filename, np.array(y))


def main():
    base_directory = "/home/loai/Images/DataTest/UC1_data"
    mtg_paths = find_rsml_paths(base_directory)
    intercepto_all: dict = {}

    # Parallélisation ici !
    with ProcessPoolExecutor(max_workers=ProcessPoolExecutor()._max_workers) as executor:
        futures = [executor.submit(compute_intercepto_for_file, path) for path in mtg_paths]
        import tqdm
        for f in tqdm.tqdm(as_completed(futures), total=len(futures)):
            mtg_box_name, res = f.result()
            intercepto_all[mtg_box_name] = res

    save_interceptos(intercepto_all)
    print("All interceptos saved.")

if __name__ == "__main__":
    main()
