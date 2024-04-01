# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
from detectron2.layers import batched_nms
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
# from detic.predictor import VisualizationDemo
# from detectron2.utils.visualizer import ColorMode, Visualizer
from collections import defaultdict
import numpy as np
from detectron2.structures import Boxes, BoxMode, pairwise_iou
import json
import os

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results

def drawbbox(img_path, bboxes, labels, out_path, render_scale=1):
    """ Plotting function to plot bboxes with labels and 2D points corresponding to 3D lidar points.
    """
    img = cv2.imread(img_path)


    for idx, bbox in enumerate(bboxes):
        cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),2)
        cv2.putText(img, labels[idx], (int(bbox[0]),int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 1)
    width, height,_ = img.shape
    pix_to_inch = 100 / render_scale
    figsize = (height / pix_to_inch, width / pix_to_inch)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(img)

    # Save to disk.
    plt.savefig(out_path, bbox_inches='tight', dpi=2.295 * pix_to_inch, pad_inches=0)
    plt.close()

    return img

def get_classnames_from_list(labels, class_mapping):
    class_names = []
    id_to_name_map = {}
    for cls_info in class_mapping:
        id_to_name_map[int(cls_info['id'])] = cls_info['name']
    
    for label in labels:
        class_names.append(id_to_name_map[label])
    
    return class_names


def inference_on_dataset_custom(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], cfg):

    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    def relabel_predictions(orig_preds, class_mapping, num_classes):
        relabeled_preds = []
        get_parent_mapping = {}
        for gt_info in class_mapping:
            if 'parent' in list(gt_info.keys()):
                get_parent_mapping[gt_info['id']] = gt_info['parent']
            else:
                get_parent_mapping[gt_info['id']] = gt_info['id']

        for pred in orig_preds:
            for i in range(len(pred['instances'])):
                pred['instances'][i]['category_id'] = get_parent_mapping[pred['instances'][i]['category_id']]
            relabeled_preds.append(pred)
        return relabeled_preds
    

    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    
    if cfg.DATASETS.RELABEL_PREDS==True:
        evaluator._predictions = relabel_predictions(deepcopy(evaluator._predictions), cfg.DATASETS.ALL_CLASSES, cfg.DATASETS.NUM_ORIG_CLASSES)
    if cfg.MODEL.ROI_HEADS.POST_LABELING_NMS_THRESH_TEST<=1.0:
        evaluator._predictions = nms_on_predictions(deepcopy(evaluator._predictions), cfg.MODEL.ROI_HEADS.POST_LABELING_NMS_THRESH_TEST, topk_per_image=cfg.TEST.DETECTIONS_PER_IMAGE, cfg=cfg)

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def nms_on_predictions(predictions, nms_thresh, topk_per_image=2500, cfg=None):
    new_predictions  = []
        
    for idx, img_pred in enumerate(predictions):   #for each image 
        print('Running NMS', idx)
        new_pred = {}
        new_pred['image_id'] = img_pred['image_id']
        new_pred['instances'] = []
        bboxes_list = [x['bbox'] for x in img_pred['instances']]
        scores_list = [x['score'] for x in img_pred['instances']]
        pred_labels_list = [x['category_id'] for x in img_pred['instances']]

        # collate boxes, scores,etc 
        bboxes = torch.Tensor(bboxes_list).view(-1,4)
        bboxes_xyxy = bboxes.clone()
        bboxes_xyxy[:,2] += bboxes_xyxy[:,0]
        bboxes_xyxy[:,3] += bboxes_xyxy[:,1]
        scores = torch.Tensor(scores_list).view(-1)
        pred_labels = torch.Tensor(pred_labels_list).view(-1)

       
        ## the case of only intra class NMS
        keep = batched_nms(bboxes_xyxy, scores, pred_labels.to(torch.int64), nms_thresh)     #bboxes in xywh form but batched nms needs it in xyxy form

        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
      
        # case when we don't want to filter predictions based on confidence threshold: helpful when we want all class predictions per bbox 
        if cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST<0.0:
            unique_boxes = bboxes[keep]                           # get unique boxes (after NMS)
            for idx, bbox in enumerate(bboxes):        # bboxes would contain box info about all classes per bbox as separate elements
                                  
                if bbox in unique_boxes:                          
                    new_img_pred = {}
                    new_img_pred['image_id'] = img_pred['instances'][idx]['image_id']
                    new_img_pred['category_id'] = int(pred_labels[idx].item())
                    new_img_pred['bbox'] = bbox.cpu().numpy().astype(float)
                    new_img_pred['score'] = float(scores[idx].item())
                    new_pred['instances'].append(new_img_pred)

        else:      
            for iidx, keep_idx in enumerate(keep):
                new_img_pred = {}
                new_img_pred['image_id'] = img_pred['instances'][keep_idx]['image_id']

                new_img_pred['category_id'] = int(pred_labels[keep_idx].item())   
                new_img_pred['bbox'] = bboxes[keep_idx].cpu().numpy().astype(float)
                new_img_pred['score'] = float(scores[keep_idx].item())

                new_pred['instances'].append(new_img_pred)
        if not new_pred['instances']:     # empty list
            continue
        else:

            new_predictions.append(new_pred)
    return new_predictions

def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results



@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
