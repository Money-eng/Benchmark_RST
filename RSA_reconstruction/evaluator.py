from __future__ import annotations

from rsml import rsml2mtg
import os
from collections import defaultdict
from typing import Dict, Optional, Sequence
import pandas as pd
from torch.nn import Module
from tqdm import tqdm
from rsml import rsml2mtg
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
        print(gt_test_folders)
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


    def evaluate(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        metric_per_box = self.metrics["per_box"]

        result_per_box: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        
        result_per_plant = defaultdict(dict) # TODO

        for split in ("val", "test"):
            for folder in tqdm(self.dict_gt[split].keys(), desc=f"Evaluating {split} folders"):
                gt = self.dict_gt[split][folder]
                pred = self.dict_pred[split][folder]

                for metric in metric_per_box:
                    name = getattr(metric, "__name__", str(metric)).split(".")[-1].split(" ")[0]
                    gt_exp  = gt["expertized"]
                    gt_bexp = gt["before_expertized"]

                    val_exp  = metric(gt_exp,  pred)   # expertisé vs. prédiction
                    val_bexp = metric(gt_bexp, pred)   # brut  vs. prédiction

                    result_per_box[split]["expertized"][name].append(val_exp)
                    result_per_box[split]["before_expertized"][name].append(val_bexp)

                # TODO : mêmes principes pour result_per_plant[...]

        import pprint
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

        # Vous pouvez convertir en DataFrame si c’est plus pratique :
        mean_per_box_dfs = {
            split: {typ: pd.Series(metrics) for typ, metrics in split_dict.items()}
            for split, split_dict in mean_per_box.items()
        }

        return {"box": mean_per_box_dfs, "plant": result_per_plant}
