from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional

import pandas as pd
from dask import delayed, compute
from dask.distributed import Client, progress
from distributed import LocalCluster
from openalea.rsml import rsml2mtg
from rsml.matching import match_plants
from rsml.misc import plant_vertices
from torch.nn import Module

from utils.misc import SEED, set_seed
from utils.mtg_operations import extract_plant_sub_mtg

set_seed(SEED)


@delayed
def _compute_metrics_for_time(
        model_name: str,
        box_name: str,
        time: int,
        mtg_pred,
        mtg_exp,
        measure: Dict[str, List[Module]],
) -> Tuple[List[dict], List[dict]]:
    rows_box, rows_plant = [], []

    # for box metrics, compute the metric on the whole mtg (num of plants)
    for func in measure.get("per_box", []):
        metric_name = getattr(func, "__name__", str(func)
                              ).split(".")[-1].split(" ")[0]
        vals = {
            "Prediction": func(mtg_pred),
            "expertized": func(mtg_exp),
        }
        rows_box.append(
            dict(
                model=model_name,
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

        matched_pred_exp, _, _ = match_plants(
            mtg_pred, mtg_exp
        )

        # put the matched plant ids (pred, exp, bexp) as keys of a dict (ex: {(42, 35, 32) : [], {28, 16, 14): []})
        matched = {}
        for (p1, p2, d) in matched_pred_exp:
            matched[(p1, p2)] = d

        failed_pred, failed_exp, failed_bexp = set(), set(), set()

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
        if failed_pred or failed_exp or failed_bexp:
            matched = {
                tuple_dist: d
                for tuple_dist, d in matched.items()
                if tuple_dist[0] not in failed_pred
                and tuple_dist[1] not in failed_exp
            }

        # make dict : (p1, p2, p3) : (pred_value, exp_value, bexp_value)
        values = {
            (p1, p2): (
                Prediction[p1],
                expertized[p2]
            )
            for (p1, p2) in matched
        }
        for (p1, p2), (pred_value, exp_value) in values.items():
            rows_plant.append(
                dict(
                    model=model_name,
                    box=box_name,
                    metric=metric_name,
                    time=time,
                    root_ids=(p1, p2),
                    source="Prediction",
                    value=pred_value,
                )
            )
            rows_plant.append(
                dict(
                    model=model_name,
                    box=box_name,
                    metric=metric_name,
                    time=time,
                    root_ids=(p1, p2),
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
            os.path.join(pred_folder, d) for d in os.listdir(pred_folder) if os.path.isdir(os.path.join(pred_folder, d))
        ]
        from os import cpu_count
        number_of_workers = int(cpu_count() * 0.9)
        threads_per_worker = 1
        cluster = LocalCluster(dashboard_address=None, n_workers=number_of_workers, threads_per_worker=threads_per_worker)
        self.client = client or Client(cluster)
        print("GT :", self.gt_folder)
        print("PRED :", self.models_folder)

    def evaluate(self) -> None:
        tasks = [] # for dask delayed tasks
        for model_folder in self.models_folder:
            model_name = os.path.basename(model_folder) # get configuration name from folder name
            # get list of rsml files for prediction and gt, for both Test and Val splits
            pred_sub = { 
                os.path.join(model_folder, d)
                for d in os.listdir(os.path.join(model_folder))
            }
            gt_sub = {
                os.path.join(self.gt_folder, d)
                for d in os.listdir(os.path.join(self.gt_folder))
            }


            dict_pred = {}
            for f in pred_sub:
                try:
                    dict_pred[os.path.basename(f)] = rsml2mtg( # os.path.basename(f) : example 230629PN006_t_0014.rsml
                        os.path.abspath(f)
                        )
                except FileNotFoundError:
                    pass
            
            dict_gt = {
                os.path.basename(f): {
                    rsml2mtg(os.path.abspath(f))
                }
                for f in gt_sub
            }

            # extract sub_mtgs for every time-step and compute metrics for each time-step with dask delayed
            for boxes in dict_gt.items():
                for b, gt in boxes: # split _ 
                    box_name, t = b.split("_t_")
                    t = int(t.split(".rsml")[0])
                    pred_full = dict_pred.get(box_name)
                    
                    exp_full = gt
                    
                    tasks.append(
                        _compute_metrics_for_time(
                            model_name,
                            box_name,
                            t,
                            pred_full,
                            exp_full,
                            self.measure,
                        )
                    )

        print(f"Launch {len(tasks)} tasks")
        print(f"Client : {self.client}")
        if isinstance(self.client, Client):
            futures = self.client.compute(tasks)
            progress(futures)
            results = self.client.gather(futures)
        else:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                results = compute(*tasks)

        rows_box, rows_plant = [], []
        for rb, rp in results:
            rows_box.extend(rb)
            rows_plant.extend(rp)

        self._save_csv(rows_box, "results_per_box.csv")
        self._save_csv(rows_plant, "results_per_plant.csv")

    def _save_csv(self, rows: List[dict], filename: str) -> None:
        if not rows:
            print(f"Rien à sauvegarder pour {filename}")
            return
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.pred_folder, filename), index=False)
        print(f"OK → {filename}")
