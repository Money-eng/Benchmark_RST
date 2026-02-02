from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Optional

import pandas as pd
from openalea.rsml import rsml2mtg
from openalea.rsml import rsml2mtg
from rsml.matching import match_plants
from torch.nn import Module
from tqdm import tqdm

from utils.misc import SEED, set_seed
from utils.mtg_operations import extract_mtg_at_time_t, extract_plant_sub_mtg

set_seed(SEED)


class ReconstructionEvaluator:
    def __init__(
            self,
            gt_folder: str,
            pred_folder: str,
            metrics: Optional[Dict[str, Module]] = None
    ) -> None:
        self.metrics = metrics

        # list folders in pred_folder
        self.models_folder = [
            os.path.join(pred_folder, folder) for folder in os.listdir(pred_folder)
        ]
        self.gt_folder = gt_folder

        print("GT folder:", self.gt_folder)
        print("Pred folder:", self.models_folder)

    def evaluate(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        # Model_name -> Val/Test -> Box_name -> Expertized/Before_expertized -> Metric_name -> Time -> Value
        result_per_box: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Dict[float, float]]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
        )
        # Model_name -> Val/Test -> Box_name -> Expertized/Before_expertized -> Metric_name -> Time -> List(Value)
        result_per_plant: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Dict[float, list]]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
        )

        # for each model sub folder in pred_folder
        for model_folder in tqdm(self.models_folder, desc="Models", total=len(self.models_folder)):
            pred_test_folders = [
                os.path.join(model_folder, "Test", folder) for folder in os.listdir(os.path.join(model_folder, "Test"))
            ]
            pred_val_folders = [
                os.path.join(model_folder, "Val", folder) for folder in os.listdir(os.path.join(model_folder, "Val"))
            ]
            gt_test_folders = [
                os.path.join(self.gt_folder, "Test", folder) for folder in
                os.listdir(os.path.join(self.gt_folder, "Test"))
            ]
            gt_val_folders = [
                os.path.join(self.gt_folder, "Val", folder) for folder in
                os.listdir(os.path.join(self.gt_folder, "Val"))
            ]
            # We assume that in every folder there is a "61_graph.rsml" file (the expertized RSML) and a "61_before_expertized_graph.rsml" file (the before expertized RSML)
            self.dict_pred = {
                "Test": {
                    folder.split('/')[-1]: rsml2mtg(os.path.join(folder,
                                                                 "61_prediction_before_expertized_graph.rsml"))
                    for folder in pred_test_folders
                },
                "Val": {
                    folder.split('/')[-1]: rsml2mtg(os.path.join(folder,
                                                                 "61_prediction_before_expertized_graph.rsml"))
                    for folder in pred_val_folders
                },
            }
            self.dict_gt = {
                "Test": {
                    folder.split('/')[-1]: {
                        "expertized": rsml2mtg(os.path.join(folder, "61_graph.rsml")),
                        "before_expertized": rsml2mtg(os.path.join(folder, "61_before_expertized_graph.rsml"))
                    }
                    for folder in gt_test_folders
                },
                "Val": {
                    folder.split('/')[-1]: {
                        "expertized": rsml2mtg(os.path.join(folder, "61_graph.rsml")),
                        "before_expertized": rsml2mtg(os.path.join(folder, "61_before_expertized_graph.rsml"))
                    }
                    for folder in gt_val_folders
                },
            }

            model_name = model_folder.split('/')[-1]
            for split in tqdm(("Val", "Test"), desc="Splits", total=2):
                print(f"Evaluating {split} split")
                for folder in tqdm(
                        self.dict_gt[split].keys(),
                        desc=f"Evaluating {split} folders",
                        leave=False
                ):
                    gt = self.dict_gt[split][folder]
                    base_pred = self.dict_pred[split][folder]
                    base_gt_exp = gt["expertized"]
                    base_gt_bexp = gt["before_expertized"]

                    # {2 : [0.0, 1.0, 2.0], 3: [0.0, 1.0, 2.0]}
                    pred_times = base_pred.property("time")
                    gt_exp_times = base_gt_exp.property("time")
                    gt_bexp_times = base_gt_bexp.property("time")

                    min_max_time = min(
                        max(max(times) for times in pred_times.values()),
                        max(max(times) for times in gt_exp_times.values()),
                        max(max(times) for times in gt_bexp_times.values())
                    )

                    for time in tqdm(
                            # TODO just last time here for first results
                            range(1, int(min_max_time) + 1),
                            desc=f"Times in {folder}",
                            leave=False
                    ):

                        pred = extract_mtg_at_time_t(base_pred, time)
                        gt_exp = extract_mtg_at_time_t(base_gt_exp, time)
                        gt_bexp = extract_mtg_at_time_t(base_gt_bexp, time)

                        print("---------------------------------- per box ----------------------------------")
                        for metric in tqdm(
                                self.metrics["per_box"],
                                desc=f"Per-box metrics t={time}",
                                leave=False
                        ):
                            name = getattr(metric, "__name__", str(
                                metric)).split(".")[-1].split(" ")[0]

                            # expertisé vs. prédiction
                            val_exp = metric(pred, gt_exp)
                            # brut  vs. prédiction
                            val_bexp = metric(pred, gt_bexp)

                            result_per_box[model_name][split]["expertized"][name][folder][time].append(
                                val_exp)
                            result_per_box[model_name][split]["before_expertized"][name][folder][time].append(
                                val_bexp)

                        print("---------------------------------- per plant ----------------------------------")
                        for metric in tqdm(
                                self.metrics["per_plant"],
                                desc=f"Per-plant metrics t={time}",
                                leave=False
                        ):
                            name = getattr(metric, "__name__", str(
                                metric)).split(".")[-1].split(" ")[0]
                            box_name = folder.split('/')[-1]

                            matched_plants_pred_exp, _, _ = match_plants(
                                pred, gt_exp)
                            matched_plants_pred_bexp, _, _ = match_plants(
                                pred, gt_bexp)

                            for plant_element in tqdm(
                                    matched_plants_pred_exp,
                                    total=len(matched_plants_pred_exp),
                                    unit=" plant",
                                    desc="  Plants (exp)",
                                    leave=False
                            ):
                                plant_id_pred = plant_element[0]
                                plant_id_gt = plant_element[1]
                                sub_mtg_pred = extract_plant_sub_mtg(
                                    pred, plant_id_pred)
                                sub_mtg_gt = extract_plant_sub_mtg(
                                    gt_exp, plant_id_gt)

                                val_exp = metric(sub_mtg_pred, sub_mtg_gt)
                                result_per_plant[model_name][split]["expertized"][name][box_name][time].append(
                                    val_exp)

                            for plant_element in tqdm(
                                    matched_plants_pred_bexp,
                                    total=len(matched_plants_pred_bexp),
                                    unit=" plant",
                                    desc="  Plants (bexp)",
                                    leave=False
                            ):
                                plant_id_pred = plant_element[0]
                                plant_id_gt = plant_element[1]
                                sub_mtg_pred = extract_plant_sub_mtg(
                                    pred, plant_id_pred)
                                sub_mtg_gt = extract_plant_sub_mtg(
                                    gt_bexp, plant_id_gt)

                                val_bexp = metric(sub_mtg_pred, sub_mtg_gt)
                                result_per_plant[model_name][split]["before_expertized"][name][box_name][time].append(
                                    val_bexp)

        self._save_results_csv(
            result_per_box, 'results_per_box.csv', is_box=True)
        self._save_results_csv(
            result_per_plant, 'results_per_plant.csv', is_box=False)
        import pprint
        pprint.pprint(result_per_plant)
        pprint.pprint(result_per_box)

    def _save_results_csv(
            self,
            data: Dict,
            filename: str,
            is_box: bool = True
    ) -> None:
        """
        Flattens the nested results dict into a DataFrame and saves as CSV.
        If is_box=True, expects structure:
          model -> split -> status -> metric -> box -> time -> [values]
        else for plants:
          model -> split -> status -> metric -> box -> time -> [values]
        """
        rows = []
        for model, splits in data.items():
            for split, statuses in splits.items():
                for status, metrics in statuses.items():
                    for metric_name, boxes in metrics.items():
                        for box_name, times in boxes.items():
                            for time, values in times.items():
                                for v in values:
                                    rows.append({
                                        'model': model,
                                        'split': split,
                                        'status': status,
                                        'metric': metric_name,
                                        'box': box_name,
                                        'time': time,
                                        'value': v
                                    })
        df = pd.DataFrame(rows)
        out_path = os.path.join(self.pred_folder, filename)
        df.to_csv(out_path, index=False)
        print(f"Saved {'box' if is_box else 'plant'} results to: {out_path}")
