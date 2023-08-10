from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmseg.registry import METRICS
from prettytable import PrettyTable

from .reachbot_metric_utils import compute_all_metrics_on_single_image


@METRICS.register_module()
class ReachbotOldMetric(BaseMetric):
    def __init__(
        self,
        sigma_factor: float = 0.02,
        threshold: float = 0.5,
        save_predictions: bool = False,
        output_dir: str = "preds",
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        """
        The metric first processes each batch of data_samples and predictions,
        and appends the processed results to the results list. Then it
        collects all results together from all ranks if distributed training
        is used. Finally, it computes the metrics of the entire dataset.
        """
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.sigma_factor = sigma_factor
        self.threshold = threshold

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_label = data_sample["pred_sem_seg"]["data"].squeeze()
            label = data_sample["gt_sem_seg"]["data"].squeeze().to(pred_label)
            self.results.append(
                compute_all_metrics_on_single_image(
                    ground_truth=label,
                    prediction_binary=pred_label,
                    sigma_factor=self.sigma_factor,
                    threshold=self.threshold,
                )
            )

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        print_log("Evaluating reachbot metrics", logger=logger)
        # Compute average and std metrics for the individual masks
        metrics_avg: Dict[str, float] = {}
        for key in results[0].keys():
            if "intersection" not in key and "union" not in key:
                metrics_avg[key] = np.mean([x[key] for x in results])

        class_table_data = PrettyTable()
        for key, val in metrics_avg.items():
            class_table_data.add_column(key, [val])

        print_log("Val metrics:", logger)
        print_log("\n" + class_table_data.get_string(), logger=logger)

        return metrics_avg
