from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Optional

import pandas as pd
from rsml import rsml2mtg
from rsml.misc import plant_vertices
from torch.nn import Module
from tqdm import tqdm

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
        # List model subfolders
        self.models_folder = [os.path.join(
            pred_folder, d) for d in os.listdir(pred_folder)]

        print("GT folder:", self.gt_folder)
        print("Pred folders:", self.models_folder)

    def evaluate(self) -> None:
        # Prepare nested dicts
        measure_per_box = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(dict)
                    )
                )
            )
        )
        measure_per_plant = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(dict)
                    )
                )
            )
        )

        for model_folder in tqdm(self.models_folder, desc="Models"):
            model_name = os.path.basename(model_folder)
            # Load pred and gt RSML -> MTG mappings
            pred_folders = {
                'Test': [os.path.join(model_folder, 'Test', d) for d in os.listdir(os.path.join(model_folder, 'Test'))],
                'Val': [os.path.join(model_folder, 'Val', d) for d in os.listdir(os.path.join(model_folder, 'Val'))]
            }
            gt_folders = {
                'Test': [os.path.join(self.gt_folder, 'Test', d) for d in
                         os.listdir(os.path.join(self.gt_folder, 'Test'))],
                'Val': [os.path.join(self.gt_folder, 'Val', d) for d in os.listdir(os.path.join(self.gt_folder, 'Val'))]
            }

            # Build dict_pred and dict_gt
            self.dict_pred = {
                split: {
                    os.path.basename(f): rsml2mtg(os.path.join(f, '61_prediction_before_expertized_graph.rsml'))
                    for f in pred_folders[split]
                }
                for split in ('Val', 'Test')
            }
            self.dict_gt = {
                split: {
                    os.path.basename(f): {
                        'expertized': rsml2mtg(os.path.join(f, '61_graph.rsml')),
                        'before_expertized': rsml2mtg(os.path.join(f, '61_before_expertized_graph.rsml'))
                    }
                    for f in gt_folders[split]
                }
                for split in ('Val', 'Test')
            }

            # Iterate splits and boxes
            for split in ('Val', 'Test'):
                for box_name, gt in tqdm(self.dict_gt[split].items(), desc=f"{model_name} {split}"):
                    pred_mtg_full = self.dict_pred[split].get(box_name)
                    if pred_mtg_full is None:
                        continue
                    exp_mtg_full = gt['expertized']
                    bexp_mtg_full = gt['before_expertized']

                    # Determine valid time range
                    times_pred = pred_mtg_full.property('time')
                    times_exp = exp_mtg_full.property('time')
                    times_bexp = bexp_mtg_full.property('time')
                    max_pred = max(max(t) for t in times_pred.values())
                    max_exp = max(max(t) for t in times_exp.values())
                    max_bexp = max(max(t) for t in times_bexp.values())
                    max_time = min(max_pred, max_exp, max_bexp)

                    for time in range(1, int(max_time) + 1):
                        mtg_pred = extract_mtg_at_time_t(pred_mtg_full, time)
                        mtg_exp = extract_mtg_at_time_t(exp_mtg_full, time)
                        mtg_bexp = extract_mtg_at_time_t(bexp_mtg_full, time)

                        # Per-box measures
                        for func in self.measure.get('per_box', []):
                            name = getattr(func, "__name__", str(
                                func)).split(".")[-1].split(" ")[0]
                            vals = {
                                'Prediction': func(mtg_pred),
                                'expertized': func(mtg_exp),
                                'before_expertized': func(mtg_bexp)
                            }
                            measure_per_box[model_name][split][box_name][name][time] = vals

                        # Per-plant measures
                        for func in self.measure.get('per_plant', []):
                            name = getattr(func, "__name__", str(
                                func)).split(".")[-1].split(" ")[0]

                            root_vertices_pred = plant_vertices(mtg_pred)
                            root_vertices_exp = plant_vertices(mtg_exp)
                            root_vertices_bexp = plant_vertices(mtg_bexp)

                            sub_mtgs_pred = {
                                v: extract_plant_sub_mtg(mtg_pred, v)
                                for v in root_vertices_pred
                            }
                            sub_mtgs_exp = {
                                v: extract_plant_sub_mtg(mtg_exp, v)
                                for v in root_vertices_exp
                            }
                            sub_mtgs_bexp = {
                                v: extract_plant_sub_mtg(mtg_bexp, v)
                                for v in root_vertices_bexp
                            }

                            measure_per_plant[model_name][split][box_name][name][time] = {
                                'Prediction': {
                                    v: func(sub_mtgs_pred[v]) for v in root_vertices_pred
                                },
                                'expertized': {
                                    v: func(sub_mtgs_exp[v]) for v in root_vertices_exp
                                },
                                'before_expertized': {
                                    v: func(sub_mtgs_bexp[v]) for v in root_vertices_bexp
                                }
                            }

        # Save CSVs
        self._save_results_csv(measure_per_box, 'results_per_box.csv')
        self._save_results_csv(measure_per_plant, 'results_per_plant.csv')

    def _save_results_csv(self, data: Dict, filename: str) -> None:
        """
        Flattens nested dict to DataFrame and saves as CSV:
        Columns: model, split, box, metric, time, <status columns>
        """
        rows = []
        for model, splits in data.items():
            for split, boxes in splits.items():
                for box, metrics in boxes.items():
                    for metric_name, times in metrics.items():
                        for time, status_vals in times.items():
                            row = {
                                'model': model,
                                'split': split,
                                'box': box,
                                'metric': metric_name,
                                'time': time
                            }
                            # Add each status as its own column
                            for status, val in status_vals.items():
                                row[status] = val
                            rows.append(row)
        if not rows:
            print(f"No data to save for {filename}")
            return
        df = pd.DataFrame(rows)
        out_path = os.path.join(self.pred_folder, filename)
        df.to_csv(out_path, index=False)
        print(f"Saved results to {out_path}")
