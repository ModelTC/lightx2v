# Standard Library
from typing import Union

# Import from third library
import torch
import torch.nn as nn
import torch.optim as optim

# Import from alt
from alt.coordinate.bbox.lidar_box3d import LiDARInstance3DBoxes
from alt.coordinate.points import Camera3DPointsTransfer
from alt.utils.file_helper import load

torch.multiprocessing.set_start_method('spawn', force=True)


class ALIGNED_FLAG:
    aligned = False
    offset = 1


class GradientOptimizeBoxes3D(nn.Module):
    def __init__(
        self,
        box3d: torch.Tensor = torch.Tensor([]),
        offset_limits: torch.Tensor = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]),
        origin: Union[tuple, list] = (0.5, 0.5, 0.5),
    ):
        super(GradientOptimizeBoxes3D, self).__init__()

        assert len(box3d) == len(offset_limits), "box3d length should be euqal to offset_limits"

        self.origin_location, self.origin_dimensions, self.origin_yaw = box3d[:3], box3d[3:6], box3d[6]
        self.offset = nn.Parameter(torch.tensor([0.0000001] * 7, dtype=torch.float, requires_grad=True))
        self.offset_limits = offset_limits
        self.origin = origin

        # self.l2_loss = nn.MSELoss()
        self.relu = nn.ReLU()

    def bbox_iou_overlaps(self, b1, b2, aligned=False, return_union=False, eps=1e-9):
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

    def get_location(self, detach=False):
        _xyz_ = self.origin_location + (self.offset[0:3].sigmoid() - 0.5) * self.offset_limits[0:3]

        if detach:
            _xyz_ = _xyz_.detach()
        return _xyz_

    def get_dimensions(self, detach=False):
        _dims_ = self.origin_dimensions + (self.offset[3:6].sigmoid() - 0.5) * self.offset_limits[3:6]

        if detach:
            _dims_ = _dims_.detach()
        return _dims_

    def get_yaws(self, detach=False):
        _yaw_ = self.origin_yaw + (self.offset[6].sigmoid() - 0.5) * self.offset_limits[6]

        if detach:
            _yaw_ = _yaw_.detach()
        return _yaw_

    def forward_optim_monocular_boxes(self, box_2d, camera_intrinsic, camera_dist, lidar2camera_rt):
        lidar_bbox3d = torch.cat((self.get_location(), self.get_dimensions(), self.get_yaws().unsqueeze(0)), dim=0).unsqueeze(0)
        lidar_bbox3d = LiDARInstance3DBoxes(lidar_bbox3d, box_dim=7, origin=self.origin)

        lidar_bbox3d_corners = lidar_bbox3d.corners

        pts_4d = torch.cat([lidar_bbox3d_corners.reshape(-1, 3), torch.ones((8, 1))], dim=-1)
        pts_camera_4d = pts_4d @ lidar2camera_rt.T
        pts_camera_3d = pts_camera_4d[:, :3]

        point_2d = Camera3DPointsTransfer.transfer_camera3d_to_image(pts_camera_3d, camera_intrinsic, camera_dist)

        x1, x2 = point_2d[:, 0].min(), point_2d[:, 0].max()
        y1, y2 = point_2d[:, 1].min(), point_2d[:, 1].max()

        # 投影框截出图像内再计算iou
        # x1 = torch.clamp(x1, 0, self.image_width)
        # x2 = torch.clamp(x2, 0, self.image_width)
        # y1 = torch.clamp(y1, 0, self.image_height)
        # y2 = torch.clamp(y2, 0, self.image_height)

        projection_bbox = torch.stack([x1, y1, x2, y2]).unsqueeze(0)
        giou, iou = self.bbox_iou_overlaps(box_2d.unsqueeze(0), projection_bbox)

        # l2_loss = self.refine self.l2_loss(box_2d.unsqueeze(0), projection_bbox)
        # pixel_loss = self.relu((box_2d.squeeze() - projection_bbox.squeeze()).abs() - 5).mean()

        loss = 1.0 - giou
        return {'giou': giou, 'iou': iou, 'loss': loss}

    @classmethod
    def refine(cls, target, calibrations, lr=0.01, epochs=400, iou_threshold=0.7):
        cur_box3d = target.location + [target.length, target.width, target.height, target.yaw]

        model = GradientOptimizeBoxes3D(torch.Tensor(cur_box3d))
        optimizer = optim.Adam(model.parameters(), lr=lr * 1.0)

        for cur_epoch in range(int(epochs)):
            optimizer.zero_grad()

            losses, ious = [], []
            if len(target.info2d) == 0:
                break

            for single_box2d in target.info2d:
                camera_intrinsic = torch.Tensor(calibrations[single_box2d.camera_name]["camera_intrinsic"])
                camera_dist = torch.Tensor(calibrations[single_box2d.camera_name]["camera_dist"])
                lidar2camera_rt = torch.Tensor(calibrations[single_box2d.camera_name]["lidar2camera_rt"])
                box_2d = torch.Tensor(single_box2d.xyxy)

                out = model.forward_optim_monocular_boxes(box_2d, camera_intrinsic, camera_dist, lidar2camera_rt)
                losses.append(out['loss'])
                ious.append(out['iou'])

            loss_mean = torch.stack(losses).mean()
            iou_mean = torch.stack(ious).mean()

            if iou_mean >= iou_threshold:
                break

            # logger.info('[Epoch {}/{}]: mean iou: {}, mean loss: {}'.format(cur_epoch + 1, epochs, iou_mean.item(), loss_mean.item()))
            loss_mean.backward()
            optimizer.step()

        return model.get_location(detach=True), model.get_dimensions(detach=True), model.get_yaws(detach=True)


if __name__ == "__main__":
    gt_targets = load("/workspace/auto-labeling-tools/result/refine/gt_targets.pkl")

    for timestamp, gt_target in gt_targets.items():
        for idx, target in enumerate(gt_target.bundle_targets_3d):
            loc, dims, yaw = GradientOptimizeBoxes3D.refine(target, gt_target.calibrations)

            gt_target.bundle_targets_3d[idx].set_location(loc.tolist())
            gt_target.bundle_targets_3d[idx].set_dimension(dims.tolist())
            gt_target.bundle_targets_3d[idx].set_rotation(yaw.item())
