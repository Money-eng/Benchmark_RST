from __future__ import annotations

import gc
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

import pandas as pd
from dask import delayed, compute
from dask.distributed import Client, as_completed
from distributed import LocalCluster
from openalea.rsml import rsml2mtg
from openalea.rsml.matching import match_plants
from openalea.rsml.misc import plant_vertices
from torch.nn import Module

from utils.misc import SEED, set_seed
from utils.mtg_operations import (
    extract_plant_sub_mtg,
)

set_seed(SEED)


@delayed
def process_box_metrics(model_name, box_name, pred_path, gt_path, gt_last_time, measure, time):
    pred_full = rsml2mtg(pred_path) # /home/loai/Documents/code/RSMLExtraction/temp/rn2_rec/DICE/HG_roots_iter_200/230629PN006_t_0000.rsml
    exp_full = rsml2mtg(gt_path)
    exp_last = rsml2mtg(gt_last_time)

    iter = int(os.path.basename(os.path.dirname(pred_path)).split("_")[-1]) # 200
    

    rows_box, rows_plant = [], []
    rb, rp = _compute_metrics(
        model_name, box_name,
        pred_full,
        exp_full,
        exp_last,
        measure,
        iter,
        time
    )
    rows_box.extend(rb)
    rows_plant.extend(rp)
    
    del pred_full
    del exp_full
    gc.collect() 
        
    return rows_box, rows_plant

def _compute_metrics(
        model_name: str,
        box_name: str,
        mtg_pred,
        mtg_exp,
        exp_last,
        measure: Dict[str, List[Module]],
        iter: int,
        time: int
) -> Tuple[List[dict], List[dict]]:
    rows_box, rows_plant = [], []

    # for box metrics, compute the metric on the whole mtg (num of plants)
    for func in measure.get("per_box", []):
        metric_name = getattr(func, "__name__", str(func)
                              ).split(".")[-1].split(" ")[0]
        vals = {
            "Prediction": func(mtg_pred),
            "expertized": func(mtg_exp)
        }
        
        # "Hourglass_" + dir.parent.parent.name
        model_n = "Hourglass"
        loss_n = os.path.basename(model_name)

        rows_box.append(
            dict(
                model=model_name,
                model_name=model_n,
                loss_name=loss_n,
                iter=iter,
                box=box_name,
                metric=metric_name,
                time=time,
                **vals,
            )
        )

    # for plant metrics, compute the metric on each plant sub-mtg and match the plants between pred, exp and bexp to compare the same plant with the same id
    for func in measure.get("per_plant", []):
        metric_name = getattr(func, "__name__", str(func)
                              ).split(".")[-1].split(" ")[0]

        sub_mtgs_pred = {
            v: extract_plant_sub_mtg(mtg_pred, v) for v in plant_vertices(mtg_pred)
        }
        sub_mtgs_exp = {
            v: extract_plant_sub_mtg(mtg_exp, v) for v in plant_vertices(mtg_exp)
        }
        v_pred = plant_vertices(mtg_pred)
        v_exp = plant_vertices(mtg_exp)
        
        if not v_pred or not v_exp:
            print(f"[SKIP] No plants found in one of the MTGs for box '{box_name}' at time {time}. Box skipped for plant metrics.")
            return rows_box, rows_plant
        
        try:
            matched_pred_exp, _, _ = match_plants(mtg_pred, mtg_exp) # {(21, 10, 0.0), (9, 5, 0.0), (1, 1, 0.0), (31, 15, 0.0), (23, 12, 0.0)}
        except ValueError as e:
            print(f"[ERROR] Error matching plants for box '{box_name}'at time {time}: {e}. Box skipped for plant metrics.")
            return rows_box, rows_plant
        
        try:
            matched_pred_exp_last_time, _, _ = match_plants(mtg_exp, exp_last)
        except ValueError as e:
            print(f"[ERROR] Error matching plants for box '{box_name}' at time {time} with last time GT: {e}. Box skipped for plant metrics.")
            return rows_box, rows_plant

        # put the matched plant ids (pred, exp) as keys of a dict (ex: {(42, 35) : [], {28, 16): []})
        matched = {}
        for (p1, p2, d) in matched_pred_exp:
            for (p3, p4, d2) in matched_pred_exp_last_time:
                if p2 == p3: 
                    matched[(p1, p2, p4)] = (d, d2) 

        failed_pred, failed_exp = set(), set()
        
        Prediction = {}
        for v in sub_mtgs_pred:
            try:
                Prediction[v] = func(sub_mtgs_pred[v])
            except Exception as e:
                print(f"[WARN] metric {metric_name} Prediction plant {v}: {e}")
                failed_pred.add(v)

        expertized = {}
        for v in sub_mtgs_exp:
            try:
                expertized[v] = func(sub_mtgs_exp[v])
            except Exception as e:
                print(f"[WARN] metric {metric_name} expertized plant {v}: {e}")
                failed_exp.add(v)

        # Retirer les matches impliquant une plante en échec
        if failed_pred or failed_exp:
            matched = {
                (p1, p2, p4): (d, d2) for (p1, p2, p4), (d, d2) in matched.items()
                if p1 not in failed_pred and p2 not in failed_exp
            }

        # make dict : (p1, p2) : (pred_value, exp_value)
        values = {
            (p1, p2, p4):  (Prediction.get(p1, None), expertized.get(p2, None)) 
            for (p1, p2, p4) in matched.keys()
        }
        for (p1, p2, p4), (pred_value, exp_value) in values.items():
            rows_plant.append(
                dict(
                    model=model_name,
                    model_name=model_n,
                    loss_name=loss_n,
                    iter=iter,
                    box=box_name,
                    metric=metric_name,
                    time=time,
                    root_ids=(p1, p2, p4),
                    source="Prediction",
                    value=pred_value,
                )
            )
            rows_plant.append(
                dict(
                    model=model_name,
                    model_name=model_n,
                    loss_name=loss_n,
                    iter=iter,
                    box=box_name,
                    metric=metric_name,
                    time=time,
                    root_ids=(p1, p2, p4),
                    source="expertized",
                    value=exp_value,
                )
            )

    return rows_box, rows_plant


class ReconstructionMesurator:
    def __init__(
            self,
            gt_folder: str,
            pred_folder: str,
            measure: Optional[Dict[str, Module]] = None,
            client: Optional[Client] = None,
    ) -> None:
        self.measure = measure or {}
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.models_folder = [
            os.path.join(pred_folder, d) 
            for d in os.listdir(pred_folder) 
            if os.path.isdir(os.path.join(pred_folder, d))
        ]
        
        total_cores = 80
        number_of_workers = total_cores
        threads_per_worker = 1
        cluster = LocalCluster(
            host='127.0.0.1',
            dashboard_address=None, 
            n_workers=number_of_workers, 
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit="8GB",
            silence_logs=50,
            local_directory="/home/lgandeel/Code/dask_tmp"
        )
        self.client = client or Client(cluster)
        print("GT :", self.gt_folder)
        print("PRED :", self.models_folder)
        print(f"Dask initialized with {number_of_workers} workers and {threads_per_worker} threads per worker.")

    def evaluate(self) -> None:
        tasks = [] # for dask delayed tasks
        
        future_measure = self.client.scatter(self.measure, broadcast=True)
        
        for model1 in self.models_folder:
            models_subdirs = os.listdir(model1)
            model_name = os.path.basename(model1) 
            
            for epoch_folder in models_subdirs:
                full_epoch_folder = os.path.join(model1, epoch_folder) 
                
                if not os.path.isdir(full_epoch_folder):
                    continue
                
                pred_path = os.path.abspath(full_epoch_folder)
                gt_path = os.path.abspath(self.gt_folder)
                
                for rsmlfile in os.listdir(pred_path): 
                    if rsmlfile.endswith(".rsml"):
                        box_name = str(os.path.basename(rsmlfile).split(".")[0].split("_")[0])
                        time = int(str(os.path.basename(rsmlfile).split(".")[0].split("_")[-1])) # ex: "T10" -> 10
                        
                        pred_path0 = os.path.abspath(os.path.join(pred_path, rsmlfile))
                        gt_path0 = os.path.abspath(os.path.join(gt_path, rsmlfile)) # 230629PN024_t_0023.rsml
                        # get time 28
                        gt_last_time = os.path.abspath(os.path.join(gt_path, rsmlfile).replace(f"_t_{time:04d}", f"_t_{28:04d}"))
                        
                        if not os.path.exists(gt_path0):
                            print(f"[WARN] GT missing for {box_name}")
                            continue

                        tasks.append(
                            process_box_metrics(
                                model_name,
                                box_name,
                                pred_path0,
                                gt_path0,
                                gt_last_time,
                                future_measure,
                                time
                            )
                        )

        print(f"Launch {len(tasks)} tasks")
        
        box_csv_path = os.path.join(self.pred_folder, "results_per_box.csv")
        plant_csv_path = os.path.join(self.pred_folder, "results_per_plant.csv")
        
        box_header_written = False
        plant_header_written = False
        
        if os.path.exists(box_csv_path): os.remove(box_csv_path)
        if os.path.exists(plant_csv_path): os.remove(plant_csv_path)
        
        if isinstance(self.client, Client):
            batch_size = 5000
            
            with tqdm(total=len(tasks), desc="Processing tasks") as pbar:
                for i in range(0, len(tasks), batch_size):
                    batch_tasks = tasks[i:i + batch_size]
                    
                    futures = self.client.compute(batch_tasks)
                    
                    for future in as_completed(futures):
                        try:
                            rb, rp = future.result() 
                            
                            if rb:
                                df_box = pd.DataFrame(rb)
                                df_box.to_csv(box_csv_path, mode='a', header=not box_header_written, index=False)
                                box_header_written = True
                                
                            if rp:
                                df_plant = pd.DataFrame(rp)
                                df_plant.to_csv(plant_csv_path, mode='a', header=not plant_header_written, index=False)
                                plant_header_written = True
                                
                        except Exception as e:
                            print(f"\n[ERREUR] Task failed : {e}")
                        finally:
                            future.release()
                        
                        pbar.update(1)                    
        else:
            from dask.diagnostics import ProgressBar
            with ProgressBar():
                results = compute(*tasks)
                rows_box, rows_plant = [], []
                for rb, rp in results:
                    rows_box.extend(rb)
                    rows_plant.extend(rp)
                self._save_csv(rows_box, "results_per_box_fallback.csv")
                self._save_csv(rows_plant, "results_per_plant_fallback.csv")

        self.client.close()
        try:
            self.client.cluster.close()
        except Exception:
            pass
        
        print("Mesuration completed.")
        
    def _save_csv(self, rows: List[dict], filename: str) -> None:
        if not rows:
            print(f"[WARN] No data to save for {filename}. CSV file will not be created.")
            return
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.pred_folder, filename), index=False)
        print(f"OK → {filename}")
