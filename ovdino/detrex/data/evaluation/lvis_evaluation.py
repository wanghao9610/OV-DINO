# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import json
import logging
import os
import pickle
from collections import OrderedDict, defaultdict

import detectron2.utils.comm as comm
import torch
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation.lvis_evaluation import _evaluate_box_proposals
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table


class LVISFixedAPEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal and instance detection/segmentation outputs using
    LVIS's metrics and evaluation API.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        topk=10000,
        fixed_ap=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                "json_file": the path to the LVIS format annotation
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
            max_dets_per_image (None or int): limit on maximum detections per image in evaluating AP
                This limit, by default of the LVIS dataset, is 300.
        """
        from lvis import LVIS

        self._logger = logging.getLogger(__name__)

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warn(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._distributed = distributed
        self._output_dir = output_dir
        self._max_dets_per_image = max_dets_per_image if max_dets_per_image else -1
        self.topk = topk
        self.fixed_ap = fixed_ap

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        self._lvis_api = LVIS(json_file)
        # Test set json files do not contain annotations (evaluation must be
        # performed using the LVIS evaluation server).
        self._do_evaluation = len(self._lvis_api.get_ann_ids()) > 0

    def reset(self):
        self._predictions = []
        self._by_cat = {}

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a LVIS model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a LVIS model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                instances = instances_to_coco_json(instances, input["image_id"])
                prediction["instances"] = instances
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)

            if self.fixed_ap:
                by_cat = defaultdict(list)
                for instance in instances:
                    by_cat[instance["category_id"]].append(instance)

                for cat, cat_anns in by_cat.items():
                    if cat not in self._by_cat:
                        self._by_cat[cat] = []

                    cur = sorted(cat_anns, key=lambda x: x["score"], reverse=True)[
                        : self.topk
                    ]
                    self._by_cat[cat] = _merge_lists(
                        self._by_cat[cat], cur, self.topk, key=lambda x: x["score"]
                    )

            self._predictions.append(prediction)

    def convert_dict_to_list(self):
        by_cat = []
        for cat, cat_anns in self._by_cat.items():
            cur = {"category_id": cat, "instances": cat_anns}
            by_cat.append(cur)

        return by_cat

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if self.fixed_ap:
                by_cat = self.convert_dict_to_list()
                by_cat = comm.gather(by_cat, dst=0)
                by_cat = list(itertools.chain(*by_cat))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            if self.fixed_ap:
                by_cat = self.convert_dict_to_list()

        if len(predictions) == 0:
            self._logger.warning("[LVISEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if self.fixed_ap:
            self._eval_by_cat(by_cat)
        elif "instances" in predictions[0]:
            self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _tasks_from_predictions(self, predictions):
        for pred in predictions:
            if "segmentation" in pred:
                return ("bbox", "segm")
        return ("bbox",)

    def _eval_by_cat(self, by_cat):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.

        Args:
            predictions (list[dict]): list of outputs from the model
        """
        results = []
        missing_dets_cats = set()

        for _by_cat in by_cat:
            cat = _by_cat["category_id"]
            cat_anns = _by_cat["instances"]
            if len(cat_anns) < self.topk:
                missing_dets_cats.add(cat)
            results.extend(
                sorted(cat_anns, key=lambda x: x["score"], reverse=True)[: self.topk]
            )
        if missing_dets_cats:
            print(
                f"\n===\n"
                f"{len(missing_dets_cats)} classes had less than {self.topk} detections!\n"
                f"Outputting {self.topk} detections for each class will improve AP further.\n"
                f"If using detectron2, please use the lvdevil/infer_topk.py script to "
                f"output a results file with {self.topk} detections for each class.\n"
                f"==="
            )

        self._logger.info("Preparing results in the LVIS format ...")
        # lvis_results = list(itertools.chain(*[x["instances"] for x in results]))
        lvis_results = results
        tasks = self._tasks or self._tasks_from_predictions(lvis_results)

        # LVIS evaluator can be used to evaluate results for COCO dataset categories.
        # In this case `_metadata` variable will have a field with COCO-specific category mapping.
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k
                for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in lvis_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]
        else:
            # unmap the category ids for LVIS (from 0-indexed to 1-indexed)
            for result in lvis_results:
                result["category_id"] += 1

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "lvis_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(lvis_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            res = _evaluate_predictions_on_lvis(
                self._lvis_api,
                lvis_results,
                task,
                max_dets_per_image=self._max_dets_per_image,
                class_names=self._metadata.get("thing_classes"),
            )
            self._results[task] = res

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.

        Args:
            predictions (list[dict]): list of outputs from the model
        """
        self._logger.info("Preparing results in the LVIS format ...")
        lvis_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(lvis_results)

        # LVIS evaluator can be used to evaluate results for COCO dataset categories.
        # In this case `_metadata` variable will have a field with COCO-specific category mapping.
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k
                for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in lvis_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]
        else:
            # unmap the category ids for LVIS (from 0-indexed to 1-indexed)
            for result in lvis_results:
                result["category_id"] += 1

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "lvis_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(lvis_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            res = _evaluate_predictions_on_lvis(
                self._lvis_api,
                lvis_results,
                task,
                max_dets_per_image=self._max_dets_per_image,
                class_names=self._metadata.get("thing_classes"),
            )
            self._results[task] = res

    def _eval_box_proposals(self, predictions):
        """
        Evaluate the box proposals in predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        if self._output_dir:
            # Saving generated box proposals to file.
            # Predicted box_proposals are in XYXY_ABS mode.
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, objectness_logits = [], [], []
            for prediction in predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
                objectness_logits.append(
                    prediction["proposals"].objectness_logits.numpy()
                )

            proposal_data = {
                "boxes": boxes,
                "objectness_logits": objectness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(
                os.path.join(self._output_dir, "box_proposals.pkl"), "wb"
            ) as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = _evaluate_box_proposals(
                    predictions, self._lvis_api, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)
        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["box_proposals"] = res


def _merge_lists(listA, listB, maxN, key):
    result = []
    indA, indB = 0, 0
    while (indA < len(listA) or indB < len(listB)) and len(result) < maxN:
        if (indB < len(listB)) and (
            indA >= len(listA) or key(listA[indA]) < key(listB[indB])
        ):
            result.append(listB[indB])
            indB += 1
        else:
            result.append(listA[indA])
            indA += 1
    return result


def _evaluate_predictions_on_lvis(
    lvis_gt, lvis_results, iou_type, max_dets_per_image=None, class_names=None
):
    """
    Args:
        iou_type (str):
        max_dets_per_image (None or int): limit on maximum detections per image in evaluating AP
            This limit, by default of the LVIS dataset, is 300.
        class_names (None or list[str]): if provided, will use it to predict
            per-category AP.

    Returns:
        a dict of {metric name: score}
    """
    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
    }[iou_type]

    logger = logging.getLogger(__name__)

    if len(lvis_results) == 0:  # TODO: check if needed
        logger.warn("No predictions from the model!")
        return {metric: float("nan") for metric in metrics}

    if iou_type == "segm":
        lvis_results = copy.deepcopy(lvis_results)
        # When evaluating mask AP, if the results contain bbox, LVIS API will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in lvis_results:
            c.pop("bbox", None)

    if max_dets_per_image is None:
        max_dets_per_image = 300  # Default for LVIS dataset

    from lvis import LVISEval, LVISResults

    logger.info(f"Evaluating with max detections per image = {max_dets_per_image}")
    lvis_results = LVISResults(lvis_gt, lvis_results, max_dets=max_dets_per_image)
    lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type)
    params = lvis_eval.params
    params.max_dets = max_dets_per_image  # No limit on detections per image.
    lvis_eval.run()
    lvis_eval.print_results()

    # Pull the standard metrics from the LVIS results
    results = lvis_eval.get_results()
    results = {metric: float(results[metric] * 100) for metric in metrics}
    logger.info(
        "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
    )
    return results
