from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Optional, Sequence

import pandas as pd
from rsml import rsml2mtg
from rsml import rsml2mtg
from rsml.matching import match_plants
from torch.nn import Module
from tqdm import tqdm

from utils.misc import SEED, set_seed
from utils.mtg_operations import extract_mtg_at_time_t, extract_plant_sub_mtg

set_seed(SEED)


class ReconstructionEvaluator:
    def __init__(
            self,
            gt_val_folders: Sequence[str],
            gt_test_folders: Sequence[str],
            pred_val_folders: Sequence[str],
            pred_test_folders: Sequence[str],
            metrics: Optional[Dict[str, Module]] = None
    ) -> None:
        self.metrics = metrics

        # We assume that in every folder there is a "61_graph.rsml" file (the expertized RSML) and a "61_before_expertized_graph.rsml" file (the before expertized RSML)

        print("GT folders:", gt_val_folders, gt_test_folders)
        print("Pred folders:", pred_val_folders, pred_test_folders)

        self.dict_pred = {
            "test": {
                folder.split('/')[-1]: rsml2mtg(os.path.join(folder, "61_prediction_before_expertized_graph.rsml"))
                for folder in pred_test_folders
            },
            "val": {
                folder.split('/')[-1]: rsml2mtg(os.path.join(folder, "61_prediction_before_expertized_graph.rsml"))
                for folder in pred_val_folders
            },
        }

        self.dict_gt = {
            "test": {
                folder.split('/')[-1]: {
                    "expertized": rsml2mtg(os.path.join(folder, "61_graph.rsml")),
                    "before_expertized": rsml2mtg(os.path.join(folder, "61_before_expertized_graph.rsml"))
                }
                for folder in gt_test_folders
            },
            "val": {
                folder.split('/')[-1]: {
                    "expertized": rsml2mtg(os.path.join(folder, "61_graph.rsml")),
                    "before_expertized": rsml2mtg(os.path.join(folder, "61_before_expertized_graph.rsml"))
                }
                for folder in gt_val_folders
            },
        }

    def evaluate(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        result_per_box: Dict[str, Dict[str, Dict[str, Dict[int, list]]]] = {
            "val": {
                "expertized": defaultdict(lambda: defaultdict(list)),
                "before_expertized": defaultdict(lambda: defaultdict(list))
            },
            "test": {
                "expertized": defaultdict(lambda: defaultdict(list)),
                "before_expertized": defaultdict(lambda: defaultdict(list))
            }
        }
        result_per_plant: Dict[str, Dict[str, Dict[str, Dict[int, list]]]] = {
            "val": {
                "expertized": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
                "before_expertized": defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            },
            "test": {
                "expertized": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
                "before_expertized": defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            }
        }

        for split in tqdm(("val", "test"), desc="Splits", total=2):
            for folder in tqdm(
                    self.dict_gt[split].keys(),
                    desc=f"Evaluating {split} folders",
                    leave=False
            ):
                gt = self.dict_gt[split][folder]
                base_pred = self.dict_pred[split][folder]
                base_gt_exp = gt["expertized"]
                base_gt_bexp = gt["before_expertized"]

                pred_times = base_pred.property("time")  # {2 : [0.0, 1.0, 2.0], 3: [0.0, 1.0, 2.0]}
                gt_exp_times = base_gt_exp.property("time")
                gt_bexp_times = base_gt_bexp.property("time")

                min_max_time = min(
                    max(max(times) for times in pred_times.values()),
                    max(max(times) for times in gt_exp_times.values()),
                    max(max(times) for times in gt_bexp_times.values())
                )

                for time in tqdm(
                        range(29, int(min_max_time) + 1),
                        desc=f"Times in {folder}",
                        leave=False
                ):

                    pred = extract_mtg_at_time_t(base_pred, time)
                    gt_exp = extract_mtg_at_time_t(base_gt_exp, time)
                    gt_bexp = extract_mtg_at_time_t(base_gt_bexp, time)

                    for metric in tqdm(
                            self.metrics["per_box"],
                            desc=f"Per-box metrics t={time}",
                            leave=False
                    ):
                        name = getattr(metric, "__name__", str(metric)).split(".")[-1].split(" ")[0]

                        val_exp = metric(pred, gt_exp)  # expertisé vs. prédiction
                        val_bexp = metric(pred, gt_bexp)  # brut  vs. prédiction

                        result_per_box[split]["expertized"][name][time].append(val_exp)
                        result_per_box[split]["before_expertized"][name][time].append(val_bexp)

                    for metric in tqdm(
                            self.metrics["per_plant"],
                            desc=f"Per-plant metrics t={time}",
                            leave=False
                    ):
                        name = getattr(metric, "__name__", str(metric)).split(".")[-1].split(" ")[0]
                        box_name = folder.split('/')[-1]

                        matched_plants_pred_exp, _, _ = match_plants(pred, gt_exp)
                        matched_plants_pred_bexp, _, _ = match_plants(pred, gt_bexp)

                        for plant_element in tqdm(
                                matched_plants_pred_exp,
                                total=len(matched_plants_pred_exp),
                                unit=" plant",
                                desc="  Plants (exp)",
                                leave=False
                        ):
                            plant_id_pred = plant_element[0]
                            plant_id_gt = plant_element[1]
                            sub_mtg_pred = extract_plant_sub_mtg(pred, plant_id_pred)
                            sub_mtg_gt = extract_plant_sub_mtg(gt_exp, plant_id_gt)

                            val_exp = metric(sub_mtg_pred, sub_mtg_gt)
                            result_per_plant[split]["expertized"][name][box_name][time].append(val_exp)

                        for plant_element in tqdm(
                                matched_plants_pred_bexp,
                                total=len(matched_plants_pred_bexp),
                                unit=" plant",
                                desc="  Plants (bexp)",
                                leave=False
                        ):
                            plant_id_pred = plant_element[0]
                            plant_id_gt = plant_element[1]
                            sub_mtg_pred = extract_plant_sub_mtg(pred, plant_id_pred)
                            sub_mtg_gt = extract_plant_sub_mtg(gt_bexp, plant_id_gt)

                            val_bexp = metric(sub_mtg_pred, sub_mtg_gt)
                            result_per_plant[split]["before_expertized"][name][box_name][time].append(val_bexp)
                print(f"Finished evaluating {split} folders.")
                print(f"Temporary results for {split} folders:")
                print("Per-box results:")
                for typ, metrics in result_per_box[split].items():
                    print(f"  {typ}:")
                    for name, times in metrics.items():
                        print(f"    {name}:")
                        for time, vals in times.items():
                            print(f"      {time}: {vals}")
                print("Per-plant results:")
                for typ, metrics in result_per_plant[split].items():
                    print(f"  {typ}:")
                    for name, boxes in metrics.items():
                        print(f"    {name}:")
                        for box_name, times in boxes.items():
                            print(f"      {box_name}:")
                            for time, vals in times.items():
                                print(f"        {time}: {vals}")

        import pprint
        pprint.pprint(result_per_plant)
        pprint.pprint(result_per_box)
        mean_per_box = {
            split: {
                typ: {
                    name: sum(vals) / len(vals) if vals else float("nan")
                    for name, vals in metrics.items()
                }
                for typ, metrics in split_dict.items()
            }
            for split, split_dict in result_per_box.items()
        }

        mean_per_plant = {
            split: {
                typ: {
                    name: {box_name: sum(vals) / len(vals) if vals else float("nan")
                           for box_name, vals in metrics.items()}
                    for name, metrics in split_dict.items()
                }
                for typ, split_dict in split_dict.items()
            }
            for split, split_dict in result_per_plant.items()
        }
        mean_per_box_dfs = {
            split: {typ: pd.Series(metrics) for typ, metrics in split_dict.items()}
            for split, split_dict in mean_per_box.items()
        }

        mean_per_plant_dfs = {
            split: {name: pd.Series(metrics) for name, metrics in split_dict.items()}
            for split, split_dict in mean_per_plant.items()
        }

        return {"box": mean_per_box_dfs, "plant": mean_per_plant_dfs}
