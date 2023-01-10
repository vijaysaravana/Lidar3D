import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
from ...utils import box_utils


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward_anchorhead(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

    def forward(self, data_dict):
        cls_preds = data_dict['batch_cls_preds']
        box_preds = data_dict['batch_box_preds']
        dir_cls_preds = data_dict['dir_cls_preds']
        #boxpreds = box_preds.numpy()
        #boxpreds.tofile("box_openPCD.bin")
        #clspreds = cls_preds.numpy()
        #clspreds.tofile("cls_openPCD.bin")
        #start_time = time.perf_counter()
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        anchors = self.anchors
        anchors = anchors[0].reshape([-1, 7])
        anchors_bv = box_utils.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])

        anchors_mask = None
        anchor_area_threshold = 1
        coors = data_dict['voxel_coords'].int()
        coors = coors[:, 1:]
        coor = coors.numpy()

        grid_size = data_dict['grid_size'].int()
        grid_size = grid_size.view(-1)
        gridsize = grid_size.numpy()

        voxel_size = data_dict['voxel_size']
        voxel_size = voxel_size.view(-1)
        voxelsize = voxel_size.numpy()

        pc_range = data_dict['point_cloud_range']
        pc_range = pc_range.view(-1)
        pcrange = pc_range.numpy()

        dense_voxel_map = box_utils.sparse_sum_for_anchors_mask(
            coor, tuple(gridsize[::-1][1:]))
        dense_voxel_map = dense_voxel_map.cumsum(0)
        dense_voxel_map = dense_voxel_map.cumsum(1)
        anchors_area = box_utils.fused_get_anchors_area(
            dense_voxel_map, anchors_bv, voxelsize, pcrange, gridsize)
        anchors_mask = anchors_area > anchor_area_threshold
        #example['anchors_mask'] = anchors_mask.astype(np.uint8)
        #example['anchors_mask'] = anchors_mask

        data_dict['anchor_mask'] = anchors_mask
        #print('AnchorHeadSingle: %.2fms' %((time.perf_counter() - start_time)*1000))

        return data_dict

