from __future__ import annotations

from rsml import rsml2mtg
import os
from collections import defaultdict
from typing import Dict, Optional, Sequence
import pandas as pd
from torch.nn import Module
from tqdm import tqdm
from rsml import rsml2mtg
from rsml.matching import match_plants
from utils.mtg_operations import extract_plant_sub_mtg
from utils.misc import SEED, set_seed

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
        # Misc. -----------------------------------------------------------
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
        result_per_box: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        result_per_plant : Dict[str, Dict[str, Dict[str, Dict[str, list]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )

        for split in ("val", "test"):
            for folder in tqdm(self.dict_gt[split].keys(), desc=f"Evaluating {split} folders"):
                gt = self.dict_gt[split][folder]
                pred = self.dict_pred[split][folder]

                for metric in self.metrics["per_box"]:
                    name = getattr(metric, "__name__", str(metric)).split(".")[-1].split(" ")[0]
                    gt_exp  = gt["expertized"]
                    gt_bexp = gt["before_expertized"]

                    val_exp  = metric(pred, gt_exp)   # expertisé vs. prédiction
                    val_bexp = metric(pred, gt_bexp)   # brut  vs. prédiction

                    result_per_box[split]["expertized"][name].append(val_exp)
                    result_per_box[split]["before_expertized"][name].append(val_bexp)

                for metric in self.metrics["per_plant"]:
                    name = getattr(metric, "__name__", str(metric)).split(".")[-1].split(" ")[0]
                    box_name = folder.split('/')[-1]
                    gt_exp  = gt["expertized"]
                    gt_bexp = gt["before_expertized"]
                    
                    matched_plants_pred_exp, _, _ = match_plants(pred, gt_exp)
                    matched_plants_pred_bexp, _, _ = match_plants(pred, gt_bexp)

                    for plant_element in matched_plants_pred_exp:
                        
                        plant_id_pred = plant_element[0]
                        plant_id_gt = plant_element[1]
                        sub_mtg_pred = extract_plant_sub_mtg(pred, plant_id_pred)
                        sub_mtg_gt = extract_plant_sub_mtg(gt_exp, plant_id_gt)

                        val_exp = metric(sub_mtg_pred, sub_mtg_gt)
                        result_per_plant[split]["expertized"][name][box_name].append(val_exp)

                    for plant_element in matched_plants_pred_bexp:
                        plant_id_pred = plant_element[0]
                        plant_id_gt = plant_element[1]
                        sub_mtg_pred = extract_plant_sub_mtg(pred, plant_id_pred)
                        sub_mtg_gt = extract_plant_sub_mtg(gt_bexp, plant_id_gt)
                        val_bexp = metric(sub_mtg_pred, sub_mtg_gt)
                        result_per_plant[split]["before_expertized"][name][box_name].append(val_bexp)

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
