"""Module for calculating IoU between predicted masks and ground truth masks."""

import torch


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    """
    Calculate the Intersection over Union (IoU) between predicted masks and ground truth masks.

    Args:
        pred_mask (torch.Tensor): Predicted mask tensor.
        gt_mask (torch.Tensor): Ground truth mask tensor.

    Returns:
        torch.Tensor: IoU values for each batch sample.

    """
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = (
        torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    )
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou
