# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import List, Tuple, Union
import torch
import os
from detectron2.layers import batched_nms, cat, move_device_like
from detectron2.structures import Boxes, Instances

logger = logging.getLogger(__name__)

def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: intersection, sized [N,M].
    """
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection


def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou

def _is_tracing():
    # (fixed in TORCH_VERSION >= 1.9)
    if torch.jit.is_scripting():
        # https://github.com/pytorch/pytorch/issues/47379
        return False
    else:
        return torch.jit.is_tracing()


def find_top_rpn_proposals(
    proposals: List[torch.Tensor],
    pred_objectness_logits: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    nms_thresh: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
    training: bool,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps for each image.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        image_sizes (list[tuple]): sizes (h, w) for each image
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_size (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        list[Instances]: list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i, sorted by their
            objectness score in descending order.
    """
    num_images = len(image_sizes)
    device = (
        proposals[0].device
        if torch.jit.is_scripting()
        else ("cpu" if torch.jit.is_tracing() else proposals[0].device)
    )

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = move_device_like(torch.arange(num_images, device=device), proposals[0])
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        Hi_Wi_A = logits_i.shape[1]
        if isinstance(Hi_Wi_A, torch.Tensor):  # it's a tensor in tracing
            num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)

        topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(
            move_device_like(
                torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device),
                proposals[0],
            )
        )

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results: List[Instances] = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_size)
        if _is_tracing() or keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]  # keep is already sorted

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results


def add_ground_truth_to_proposals(
    gt: Union[List[Instances], List[Boxes]], proposals: List[Instances]
) -> List[Instances]:
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.

    Args:
        gt(Union[List[Instances], List[Boxes]): list of N elements. Element i is a Instances
            representing the ground-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert gt is not None

    if len(proposals) != len(gt):
        raise ValueError("proposals and gt should have the same length as the number of images!")
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image(gt_i, proposals_i)
        for gt_i, proposals_i in zip(gt, proposals)
    ]

def get_gt_proposals(
    gt: Union[List[Instances], List[Boxes]], proposals: List[Instances]
) -> List[Instances]:
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.

    Args:
        gt(Union[List[Instances], List[Boxes]): list of N elements. Element i is a Instances
            representing the ground-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert gt is not None

    if len(proposals) != len(gt):
        raise ValueError("proposals and gt should have the same length as the number of images!")
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image2(gt_i, proposals_i)
        for gt_i, proposals_i in zip(gt, proposals)
    ]


def add_ground_truth_to_proposals_single_image(
    gt: Union[Instances, Boxes], proposals: Instances
) -> Instances:
    """
    Augment `proposals` with `gt`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with gt and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    if isinstance(gt, Boxes):
        # convert Boxes to Instances
        gt = Instances(proposals.image_size, gt_boxes=gt)
    gt_boxes = gt.gt_boxes
    device = proposals.objectness_logits.device
    # Assign all ground-truth boxes an objectness logit corresponding to
    # P(object) = sigmoid(logit) =~ 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)

    # Concatenating gt_boxes with proposals requires them to have the same fields
    gt_proposal = Instances(proposals.image_size, **gt.get_fields())
    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits

    for key in proposals.get_fields().keys():
        assert gt_proposal.has(
            key
        ), "The attribute '{}' in `proposals` does not exist in `gt`".format(key)

    # NOTE: Instances.cat only use fields from the first item. Extra fields in latter items
    # will be thrown away.
    new_proposals = Instances.cat([proposals, gt_proposal])

    return new_proposals

def update_targets_with_pseudolabels(targets, file_names, pseudolabel_path, confidence_thresh, iou_thresh=0.5, one_class=False, one_class_id=-1):

    for idx, target_info in enumerate(targets):
        img_path = file_names[idx]
        pseudolabel_file = os.path.join(pseudolabel_path, os.path.basename(img_path).replace('.jpg', '.pth'))
        device = targets[idx]._fields['gt_classes'].device

        pseudolabels = torch.load(pseudolabel_file, map_location=device)
        selected_inds = torch.where(pseudolabels['scores']>=confidence_thresh)[0]
        new_boxes = pseudolabels['bboxs'][selected_inds]
        new_labels = pseudolabels['pred_labels'][selected_inds].long()

        tgt_pseudolabels_pairwise_iou = pairwise_iou(targets[idx]._fields['gt_boxes'], Boxes(new_boxes))    # GT x Num Boxes
        maxiou_overlap_gt_new_boxes = torch.max(tgt_pseudolabels_pairwise_iou, dim=0)[0]   # take max across GT => if a box has overlap >iou_thresh with any GT, then no need of adding it to targets
        unique_box_inds = torch.where(maxiou_overlap_gt_new_boxes<iou_thresh)[0]
        unique_boxes = new_boxes[unique_box_inds]
        unique_box_labels = new_labels[unique_box_inds] 
        if one_class:
            new_unique_box_labels = torch.ones_like(unique_box_labels)*one_class_id
            targets[idx]._fields['gt_classes'] = torch.cat([targets[idx]._fields['gt_classes'], new_unique_box_labels])

        else:
            targets[idx]._fields['gt_classes'] = torch.cat([targets[idx]._fields['gt_classes'], unique_box_labels])
        targets[idx]._fields['gt_boxes'].tensor = torch.cat([targets[idx]._fields['gt_boxes'].tensor, unique_boxes])

    return targets

    

def add_ground_truth_to_proposals_single_image2(
    gt: Union[Instances, Boxes], proposals: Instances
) -> Instances:
    """
    Augment `proposals` with `gt`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with gt and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    if isinstance(gt, Boxes):
        # convert Boxes to Instances
        gt = Instances(proposals.image_size, gt_boxes=gt)
    gt_boxes = gt.gt_boxes
    device = proposals.objectness_logits.device
    # Assign all ground-truth boxes an objectness logit corresponding to
    # P(object) = sigmoid(logit) =~ 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)

    # Concatenating gt_boxes with proposals requires them to have the same fields
    gt_proposal = Instances(proposals.image_size, **gt.get_fields())
    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits
    gt_proposal.scores = gt_logits
    gt_proposal.pred_classes = gt.gt_classes


    for key in proposals.get_fields().keys():
        assert gt_proposal.has(
            key
        ), "The attribute '{}' in `proposals` does not exist in `gt`".format(key)

    # NOTE: Instances.cat only use fields from the first item. Extra fields in latter items
    # will be thrown away.
    # new_proposals = Instances.cat([proposals, gt_proposal])
    return gt_proposal
