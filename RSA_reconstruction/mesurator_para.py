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
from utils.mtg_operations import (
    extract_mtg_at_time_t,
    extract_plant_sub_mtg,
)

set_seed(SEED)


@delayed
def _compute_metrics_for_time(
        model_name: str,
        split: str,
        box_name: str,
        time: int,
        mtg_pred,
        mtg_exp,
        mtg_bexp,
        measure: Dict[str, List[Module]],
) -> Tuple[List[dict], List[dict]]:
    rows_box, rows_plant = [], []

    # -- métriques par boîte --------------------------------------
    for func in measure.get("per_box", []):
        metric_name = getattr(func, "__name__", str(func)
                              ).split(".")[-1].split(" ")[0]
        vals = {
            "Prediction": func(mtg_pred),
            "expertized": func(mtg_exp),
            "before_expertized": func(mtg_bexp),
        }
        rows_box.append(
            dict(
                model=model_name,
                split=split,
                box=box_name,
                metric=metric_name,
                time=time,
                **vals,
            )
        )

    # -- métriques par plante -------------------------------------
    for func in measure.get("per_plant", []):
        metric_name = getattr(func, "__name__", str(func)
                              ).split(".")[-1].split(" ")[0]

        sub_mtgs_pred = {
            v: extract_plant_sub_mtg(mtg_pred, v) for v in plant_vertices(mtg_pred)
        }
        sub_mtgs_exp = {
            v: extract_plant_sub_mtg(mtg_exp, v) for v in plant_vertices(mtg_exp)
        }
        sub_mtgs_bexp = {
            v: extract_plant_sub_mtg(mtg_bexp, v) for v in plant_vertices(mtg_bexp)
        }

        matched_pred_exp, _, _ = match_plants(
            # {(21, 10, 0.0), (9, 5, 0.0), (1, 1, 0.0), (31, 15, 0.0), (23, 12, 0.0)}
            mtg_pred, mtg_exp
        )
        matched_exp_bexp, _, _ = match_plants(
            # {(10, 15, 0.0), (5, 12, 0.0), (1, 1, 0.0), (15, 14, 0.0), (12, 60, 0.0)}
            mtg_exp, mtg_bexp
        )

        # put the matched plant ids (pred, exp, bexp) as keys of a dict (ex: {(42, 35, 32) : [], {28, 16, 14): []})
        matched = {}
        for (p1, p2, d) in matched_pred_exp:
            for (p3, p4, d2) in matched_exp_bexp:
                if p2 == p3:
                    matched[(p1, p2, p4)] = (d, d2)

        # Calcul des métriques par plante avec gestion des exceptions
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

        before_expertized = {}
        for v in sub_mtgs_bexp:
            try:
                before_expertized[v] = func(sub_mtgs_bexp[v])
            except Exception as e:
                print(f"[WARN] metric {metric_name} before_expertized plant {v}: {e}")
                failed_bexp.add(v)

        # Retirer les matches impliquant une plante en échec
        if failed_pred or failed_exp or failed_bexp:
            matched = {
                triple: dists
                for triple, dists in matched.items()
                if triple[0] not in failed_pred
                and triple[1] not in failed_exp
                and triple[2] not in failed_bexp
            }

        # make dict : (p1, p2, p3) : (pred_value, exp_value, bexp_value)
        values = {
            (p1, p2, p3): (
                Prediction[p1],
                expertized[p2],
                before_expertized[p3],
            )
            for (p1, p2, p3) in matched
        }
        for (p1, p2, p3), (pred_value, exp_value, bexp_value) in values.items():
            rows_plant.append(
                dict(
                    model=model_name,
                    split=split,
                    box=box_name,
                    metric=metric_name,
                    time=time,
                    root_ids=(p1, p2, p3),
                    source="Prediction",
                    value=pred_value,
                )
            )
            rows_plant.append(
                dict(
                    model=model_name,
                    split=split,
                    box=box_name,
                    metric=metric_name,
                    time=time,
                    root_ids=(p1, p2, p3),
                    source="expertized",
                    value=exp_value,
                )
            )
            rows_plant.append(
                dict(
                    model=model_name,
                    split=split,
                    box=box_name,
                    metric=metric_name,
                    time=time,
                    root_ids=(p1, p2, p3),
                    source="before_expertized",
                    value=bexp_value,
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
        tasks = []
        for model_folder in self.models_folder:
            model_name = os.path.basename(model_folder)
            pred_sub = {
                s: [
                    os.path.join(model_folder, s, d)
                    for d in os.listdir(os.path.join(model_folder, s))
                ]
                for s in ("Test", "Val")
            }
            gt_sub = {
                s: [
                    os.path.join(self.gt_folder, s, d)
                    for d in os.listdir(os.path.join(self.gt_folder, s))
                ]
                for s in ("Test", "Val")
            }


            dict_pred = {}
            dict_pred = {s: {} for s in ("Val", "Test")}
            for s in ("Val", "Test"):
                for f in pred_sub[s]:
                    try:
                        dict_pred[s][os.path.basename(f)] = rsml2mtg(
                            os.path.abspath(os.path.join(
                                f, "61_prediction_before_expertized_graph.rsml")
                            ))
                    except FileNotFoundError:
                        pass
            
            dict_gt = {
                s: {
                    os.path.basename(f): {
                        "expertized": rsml2mtg(os.path.abspath(os.path.join(f, "61_graph.rsml"))),
                        "before_expertized": rsml2mtg(
                            os.path.abspath(os.path.join(f, "61_before_expertized_graph.rsml"))
                        ),
                    }
                    for f in gt_sub[s]
                }
                for s in ("Val", "Test")
            }

            for split, boxes in dict_gt.items():
                for box_name, gt in boxes.items():
                    pred_full = dict_pred[split].get(box_name)
                    if pred_full is None:
                        continue
                    exp_full = gt["expertized"]
                    bexp_full = gt["before_expertized"]

                    max_time = min(
                        max(max(t)
                            for t in pred_full.property("time").values()),
                        max(max(t)
                            for t in exp_full.property("time").values()),
                        max(max(t)
                            for t in bexp_full.property("time").values()),
                    )
                    for t in range(1, int(max_time) + 1):
                        tasks.append(
                            _compute_metrics_for_time(
                                model_name,
                                split,
                                box_name,
                                t,
                                extract_mtg_at_time_t(pred_full, t),
                                extract_mtg_at_time_t(exp_full, t),
                                extract_mtg_at_time_t(bexp_full, t),
                                self.measure,
                            )
                        )

        print(f"Lancement de {len(tasks)} tâches…")
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
