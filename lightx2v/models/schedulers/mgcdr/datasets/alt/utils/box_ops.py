# Standard Library
import copy

# Import from third library
import cv2
import numpy as np
import torch


class ALIGNED_FLAG:
    aligned = False
    offset = 1


def bbox_iou_overlaps(b1, b2, aligned=False, return_union=False, eps=1e-9):
    """
    Arguments:
        b1: dts, [n, >=4] (x1, y1, x2, y2, ...)
        b1: gts, [n, >=4] (x1, y1, x2, y2, ...)

    Returns:
        intersection-over-union pair-wise.
    """
    area1 = (b1[:, 2] - b1[:, 0] + ALIGNED_FLAG.offset) * (b1[:, 3] - b1[:, 1] + ALIGNED_FLAG.offset)
    area2 = (b2[:, 2] - b2[:, 0] + ALIGNED_FLAG.offset) * (b2[:, 3] - b2[:, 1] + ALIGNED_FLAG.offset)
    # only for giou loss
    lt1 = torch.max(b1[:, :2], b2[:, :2])
    rb1 = torch.max(b1[:, 2:4], b2[:, 2:4])
    lt2 = torch.min(b1[:, :2], b2[:, :2])
    rb2 = torch.min(b1[:, 2:4], b2[:, 2:4])
    # en = (lt1 < rb2).type(lt1.type()).prod(dim=1)
    # inter_area = inter_area * en
    wh1 = (rb2 - lt1 + ALIGNED_FLAG.offset).clamp(min=0)
    wh2 = (rb1 - lt2 + ALIGNED_FLAG.offset).clamp(min=0)
    inter_area = wh1[:, 0] * wh1[:, 1]
    union_area = area1 + area2 - inter_area

    iou = inter_area / torch.clamp(union_area, min=1)

    ac_union = wh2[:, 0] * wh2[:, 1] + eps
    giou = iou - (ac_union - union_area) / ac_union
    return giou, iou


def compute_lines_mask_ratio(ground_masks: list, line_points: list):
    """
    Infer whether line is visible in image.

    :param ground_mask_tensor: Mask tensor for ground in image.
    :param line_points: 2D line points in image.
    """
    if not ground_masks:
        return 0.8

    if len(line_points) < 2:
        return 0

    union_mask = ground_masks[0].squeeze()
    for array in ground_masks[1:]:
        union_mask = np.logical_or(union_mask, array.squeeze())

    in_mask_points, in_image_points = [], []
    for line_point in line_points:
        x, y = int(line_point[0]), int(line_point[1])
        row, col = union_mask.shape
        if x >= col or y >= row or x < 0 or y < 0:
            continue
        in_image_points.append([x, y])

        if union_mask[y][x]:
            in_mask_points.append([x, y])

    return len(in_mask_points) / len(in_image_points)


def compute_polygon_mask_ratio(masks, polygon, shape):
    """
    计算多边形与 mask 的占比。
    """
    if len(masks) == 0:
        return 0.8

    if len(polygon) == 0:
        return 1

    union_mask = masks[0].squeeze()
    for array in masks[1:]:
        union_mask = np.logical_or(union_mask, array.squeeze())

    extend_shape = [shape[0] * 3, shape[1] * 3]
    extend_polygon = copy.deepcopy(polygon)
    extend_polygon[:, 0] += shape[1]
    extend_polygon[:, 1] += shape[0]

    extend_polygon_mask = np.zeros(extend_shape, dtype=np.uint8)
    extend_polygon_mask = cv2.fillPoly(extend_polygon_mask, [np.array(extend_polygon, dtype=np.int32)], 1)

    polygon_mask = extend_polygon_mask[shape[0] : -shape[0], shape[1] : -shape[1]]

    # polygon_mask = np.zeros(shape, dtype=np.uint8)
    # polygon_mask = cv2.fillPoly(polygon_mask, [np.array(polygon, dtype=np.int32)], 1)

    intersection = union_mask & polygon_mask
    intersection_area = intersection.sum().item()
    polygon_area = polygon_mask.sum().item()

    if polygon_area == 0:
        return 0
    return intersection_area / polygon_area
