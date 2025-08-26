from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional

import pandas as pd
from dask import delayed, compute
from dask.distributed import Client, progress
from distributed import LocalCluster
from rsml import rsml2mtg
from rsml.matching import match_plants
from rsml.misc import plant_vertices
from torch.nn import Module

from utils.misc import SEED, set_seed
from utils.mtg_operations import (
    extract_mtg_at_time_t,
    extract_plant_sub_mtg,
)

set_seed(SEED)


# ================================================================
# 1)  MESURATOR (inchangé depuis la version précédente)            
# ================================================================
# la classe et la fonction delayed ont été légèrement renommées pour éviter les collisions
# ------------------------------------------------------------------
@delayed
def _mesurator_metrics_for_time(
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

    for func in measure.get("per_box", []):
        metric_name = getattr(func, "__name__", str(func)).split(".")[-1].split(" ")[0]
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

    for func in measure.get("per_plant", []):
        metric_name = getattr(func, "__name__", str(func)).split(".")[-1].split(" ")[0]

        sub_mtgs_pred = {v: extract_plant_sub_mtg(mtg_pred, v) for v in plant_vertices(mtg_pred)}
        sub_mtgs_exp = {v: extract_plant_sub_mtg(mtg_exp, v) for v in plant_vertices(mtg_exp)}
        sub_mtgs_bexp = {v: extract_plant_sub_mtg(mtg_bexp, v) for v in plant_vertices(mtg_bexp)}

        rows_plant.append(
            dict(
                model=model_name,
                split=split,
                box=box_name,
                metric=metric_name,
                time=time,
                Prediction={v: func(sub_mtgs_pred[v]) for v in sub_mtgs_pred},
                expertized={v: func(sub_mtgs_exp[v]) for v in sub_mtgs_exp},
                before_expertized={v: func(sub_mtgs_bexp[v]) for v in sub_mtgs_bexp},
            )
        )

    return rows_box, rows_plant


class ReconstructionMesurator:
    """Version parallèle (inchangée) – voir la description de la PR précédente"""

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
        self.models_folder = [os.path.join(pred_folder, d)
                              for d in os.listdir(pred_folder)
                              if os.path.isdir(os.path.join(pred_folder, d))]
        cluster = LocalCluster(dashboard_address=":8787")
        self.client = client or Client(cluster)
        print("GT  :", self.gt_folder)
        print("PRED:", self.models_folder)
        print("Dashboard →", self.client.dashboard_link)

    # -- méthode evaluate identique à la précédente (omise pour concision) --
    def evaluate(self) -> None:  # ← contenu inchangé par rapport à la version précédente
        pass  # placeholder : contenu complet disponible dans la PR originale


# ================================================================
# 2)  EVALUATOR (nouvelle version parallélisée)                    
# ================================================================
@delayed
def _evaluator_metrics_for_time(
        model_name: str,
        split: str,
        box_name: str,
        time: int,
        mtg_pred,
        mtg_gt_exp,
        mtg_gt_bexp,
        metrics: Dict[str, List[Module]],
) -> Tuple[List[dict], List[dict]]:
    """Calcule toutes les métriques pour (model,split,box,time)."""

    rows_box, rows_plant = [], []

    # ---------- métriques par boîte ------------------------------
    for metric in metrics.get("per_box", []):
        name = getattr(metric, "__name__", str(metric)).split(".")[-1].split(" ")[0]
        rows_box.extend([
            dict(model=model_name, split=split, status="expertized", metric=name, box=box_name, time=time,
                 value=metric(mtg_pred, mtg_gt_exp)),
            dict(model=model_name, split=split, status="before_expertized", metric=name, box=box_name, time=time,
                 value=metric(mtg_pred, mtg_gt_bexp)),
        ])

    # ---------- métriques par plante -----------------------------
    for metric in metrics.get("per_plant", []):
        name = getattr(metric, "__name__", str(metric)).split(".")[-1].split(" ")[0]

        matched_exp, _, _ = match_plants(mtg_pred, mtg_gt_exp)
        matched_bexp, _, _ = match_plants(mtg_pred, mtg_gt_bexp)

        # — expertized
        for plant_pred, plant_gt, _ in matched_exp:
            val = metric(
                extract_plant_sub_mtg(mtg_pred, plant_pred),
                extract_plant_sub_mtg(mtg_gt_exp, plant_gt),
            )
            rows_plant.append(dict(
                model=model_name,
                split=split,
                status="expertized",
                metric=name,
                box=box_name,
                time=time,
                value=val,
            ))
        # — before_expertized
        for plant_pred, plant_gt, _ in matched_bexp:
            val = metric(
                extract_plant_sub_mtg(mtg_pred, plant_pred),
                extract_plant_sub_mtg(mtg_gt_bexp, plant_gt),
            )
            rows_plant.append(dict(
                model=model_name,
                split=split,
                status="before_expertized",
                metric=name,
                box=box_name,
                time=time,
                value=val,
            ))

    return rows_box, rows_plant


class ReconstructionEvaluator:
    """Évaluation parallèle (boîte + plante) avec barre de progression."""

    def __init__(
            self,
            gt_folder: str,
            pred_folder: str,
            metrics: Optional[Dict[str, Module]] = None,
            client: Optional[Client] = None,
    ) -> None:
        self.metrics = metrics or {}
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.models_folder = [
            os.path.join(pred_folder, d)
            for d in os.listdir(pred_folder)
            if os.path.isdir(os.path.join(pred_folder, d))
        ]
        self.client = client or Client()
        print("GT  :", self.gt_folder)
        print("PRED:", self.models_folder)
        print("Dashboard →", self.client.dashboard_link)

    # --------------------------------------------------------------
    def evaluate(self) -> None:
        tasks = []

        for model_folder in self.models_folder:
            model_name = os.path.basename(model_folder)

            pred_sub = {
                s: [os.path.join(model_folder, s, d) for d in os.listdir(os.path.join(model_folder, s))]
                for s in ("Test", "Val") if os.path.isdir(os.path.join(model_folder, s))
            }
            gt_sub = {
                s: [os.path.join(self.gt_folder, s, d) for d in os.listdir(os.path.join(self.gt_folder, s))]
                for s in ("Test", "Val") if os.path.isdir(os.path.join(self.gt_folder, s))
            }

            dict_pred = {
                s: {
                    os.path.basename(f): rsml2mtg(os.path.join(f, "61_prediction_before_expertized_graph.rsml"))
                    for f in pred_sub[s]
                } for s in ("Val", "Test")
            }
            dict_gt = {
                s: {
                    os.path.basename(f): {
                        "expertized": rsml2mtg(os.path.join(f, "61_graph.rsml")),
                        "before_expertized": rsml2mtg(os.path.join(f, "61_before_expertized_graph.rsml")),
                    } for f in gt_sub[s]
                } for s in ("Val", "Test")
            }

            for split, boxes in dict_gt.items():
                for box_name, gt in boxes.items():
                    pred_full = dict_pred[split].get(box_name)
                    if pred_full is None:
                        continue
                    gt_exp_full = gt["expertized"]
                    gt_bexp_full = gt["before_expertized"]

                    max_time = min(
                        max(max(t) for t in pred_full.property("time").values()),
                        max(max(t) for t in gt_exp_full.property("time").values()),
                        max(max(t) for t in gt_bexp_full.property("time").values()),
                    )

                    for t in range(1, int(max_time) + 1):
                        tasks.append(
                            _evaluator_metrics_for_time(
                                model_name,
                                split,
                                box_name,
                                t,
                                extract_mtg_at_time_t(pred_full, t),
                                extract_mtg_at_time_t(gt_exp_full, t),
                                extract_mtg_at_time_t(gt_bexp_full, t),
                                self.metrics,
                            )
                        )

        # ----------------------------------------------------------
        print(f"\n→ Lancement de {len(tasks)} tâches Dask (Evaluator)…")
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

        self._save_csv(rows_box, "results_per_box_evaluator.csv")
        self._save_csv(rows_plant, "results_per_plant_evaluator.csv")

    # --------------------------------------------------------------
    def _save_csv(self, rows: List[dict], filename: str) -> None:
        if not rows:
            print(f"Rien à sauvegarder pour {filename}")
            return
        df = pd.DataFrame(rows)
        out = os.path.join(self.pred_folder, filename)
        df.to_csv(out, index=False)
        print("Sauvé →", out)
