from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Optional, Tuple

import pandas as pd
from rsml import rsml2mtg
from rsml.misc import plant_vertices
from rsml.matching import match_plants
from torch.nn import Module
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.misc import SEED, set_seed
from utils.mtg_operations import extract_mtg_at_time_t, extract_plant_sub_mtg

set_seed(SEED)


class ReconstructionMesurator:
    def __init__(
        self,
        gt_folder: str,
        pred_folder: str,
        measure: Optional[Dict[str, Module]] = None
    ) -> None:
        self.measure = measure or {}
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.models_folder = [os.path.join(pred_folder, d) for d in os.listdir(pred_folder)]
        print("GT folder:", self.gt_folder)
        print("Pred folders:", self.models_folder)

    def _compute_box(
        self,
        model_folder: str,
        split: str,
        box_name: str
    ) -> Tuple[Dict, Dict]:
        # Load MTGs
        pred_path = os.path.join(model_folder, split, box_name, '61_prediction_before_expertized_graph.rsml')
        gt_base = os.path.join(self.gt_folder, split, box_name)
        mtg_pred_full = rsml2mtg(pred_path)
        mtg_exp_full = rsml2mtg(os.path.join(gt_base, '61_graph.rsml'))
        mtg_bexp_full = rsml2mtg(os.path.join(gt_base, '61_before_expertized_graph.rsml'))

        # Determine time range
        times_pred = mtg_pred_full.property('time')
        times_exp = mtg_exp_full.property('time')
        times_bexp = mtg_bexp_full.property('time')
        max_time = int(min(
            max(max(t) for t in times_pred.values()),
            max(max(t) for t in times_exp.values()),
            max(max(t) for t in times_bexp.values())
        ))

        box_results: Dict[int, Dict[str, float]] = {}
        plant_results: Dict[int, Dict[str, Dict[int, float]]] = {}

        for time in range(1, max_time + 1):
            mtg_pred = extract_mtg_at_time_t(mtg_pred_full, time)
            mtg_exp = extract_mtg_at_time_t(mtg_exp_full, time)
            mtg_bexp = extract_mtg_at_time_t(mtg_bexp_full, time)

            # per-box
            box_vals: Dict[str, Dict[str, float]] = {}
            for func in self.measure.get('per_box', []):
                name = getattr(func, "__name__", str(
                                func)).split(".")[-1].split(" ")[0]
                box_vals[name] = {
                    'Prediction': func(mtg_pred),
                    'expertized': func(mtg_exp),
                    'before_expertized': func(mtg_bexp)
                }

            # per-plant
            plant_vals: Dict[str, Dict[int, float]] = {}
            for func in self.measure.get('per_plant', []):
                name = getattr(func, "__name__", str(
                                func)).split(".")[-1].split(" ")[0]
                roots_pred = plant_vertices(mtg_pred)
                roots_exp = plant_vertices(mtg_exp)
                roots_bexp = plant_vertices(mtg_bexp)

                # predictions
                sub_pred = {v: extract_plant_sub_mtg(mtg_pred, v) for v in roots_pred}
                sub_exp = {v: extract_plant_sub_mtg(mtg_exp, v) for v in roots_exp}
                sub_bexp = {v: extract_plant_sub_mtg(mtg_bexp, v) for v in roots_bexp}

                plant_vals[name] = {
                    'Prediction': {v: func(sub_pred[v]) for v in sub_pred},
                    'expertized': {v: func(sub_exp[v]) for v in sub_exp},
                    'before_expertized': {v: func(sub_bexp[v]) for v in sub_bexp}
                }

            box_results[time] = box_vals
            plant_results[time] = plant_vals

        return box_results, plant_results

    def evaluate(self) -> None:
        measure_per_box = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )  # model -> split -> box -> results
        measure_per_plant = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )

        tasks = []
        for model_folder in self.models_folder:
            model_name = os.path.basename(model_folder)
            for split in ('Val', 'Test'):
                boxes = os.listdir(os.path.join(model_folder, split))
                for box_name in boxes:
                    tasks.append((model_folder, model_name, split, box_name))

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {
                executor.submit(self._compute_box, mf, split, box): (model, split, box)
                for mf, model, split, box in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Boxes"):
                model, split, box = futures[future]
                box_res, plant_res = future.result()
                measure_per_box[model][split][box] = box_res
                measure_per_plant[model][split][box] = plant_res

        self._save_results_csv(measure_per_box, 'results_per_box.csv')
        self._save_results_csv(measure_per_plant, 'results_per_plant.csv')

    def _save_results_csv(self, data: Dict, filename: str) -> None:
        rows = []
        for model, splits in data.items():
            for split, boxes in splits.items():
                for box, times in boxes.items():
                    for time, vals in times.items():
                        row = {'model': model, 'split': split, 'box': box, 'time': time}
                        # flatten status or metric/value
                        # detect if vals is per-box (status dict) or per-plant (metric dict)
                        if all(isinstance(v, dict) for v in vals.values()):
                            # per-box: vals is metric->status->value
                            for metric, status_dict in vals.items():
                                for status, value in status_dict.items():
                                    rows.append({**row, 'metric': metric, 'status': status, 'value': value})
                        else:
                            # per-plant: vals is metric->root->value
                            metric = time  # placeholder, should not occur here
                            for root, value in vals.items():
                                rows.append({**row, 'metric': metric, 'root': root, 'value': value})
        df = pd.DataFrame(rows)
        out_path = os.path.join(self.pred_folder, filename)
        df.to_csv(out_path, index=False)
        print(f"Saved results to {out_path}")
