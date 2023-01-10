import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from sys import getsizeof
from .vfe_template import VFETemplate
import threading
from pathlib import Path


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, openvino_ie):
        super().__init__(model_cfg=model_cfg)
        # print(f'Available devices :: {openvino_ie.get_available_devices()}')
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        # ie = IECore()
        # self.ie = ie
        # self.ie = openvino_ie
        model_file_pfe = str(Path(__file__).resolve().parents[4] / 'tools'/'pfe.xml')
        model_weight_pfe = str(Path(__file__).resolve().parents[4] / 'tools'/'pfe.bin')
        print(f'Model file : {model_file_pfe}')
        print(f'model_weight_pfe  : {model_weight_pfe}')
        self.net_pfe = openvino_ie.read_network(model_file_pfe, model_weight_pfe)

        myriad1_config = {}
        myriad2_config = {}

        # openvino_ie.set_config(config=myriad1_config, device_name="MYRIAD.1.1.4-ma2480")
        # openvino_ie.set_config(config=myriad2_config, device_name="MYRIAD.1.1.1-ma2480")

        self.exec_net_pfe = openvino_ie.load_network(network=self.net_pfe, device_name="CPU")
        # Load the network to the multi-device, specifying the priorities
        # self.exec_net_pfe = openvino_ie.load_network(
        #     network=self.net_pfe, device_name="MULTI", num_requests=8, config={"MULTI_DEVICE_PRIORITIES": "MYRIAD.1.1.4-ma2480,MYRIAD.1.1.1-ma2480"}
        # )
        # Query the optimal number of requests
        nireq = self.exec_net_pfe.get_metric("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        print(f'Optimal number of Infer req PFE:: {nireq}')
        
        self.frame_id = 0
        self.event = threading.Event()
        self.queue = []


    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def preprocessing(self, batch_dict, **kwargs):
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        #start_time = time.perf_counter()

        coors_x = coords[:, 3].float()
        coors_y = coords[:, 2].float()
        x_sub = coors_x.unsqueeze(1) * 0.16 + 0.1
        y_sub = coors_y.unsqueeze(1) * 0.16 + -39.9
        # ones = torch.ones([1, 100], dtype=torch.float32, device="vpu")
        ones = torch.ones([1, 100], dtype=torch.float32, device="cpu")
        x_sub_shaped = torch.mm(x_sub, ones).unsqueeze(0).unsqueeze(0)
        y_sub_shaped = torch.mm(y_sub, ones).unsqueeze(0).unsqueeze(0)

        voxel_count = voxel_features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, 0).type_as(voxel_features)
        mask = torch.unsqueeze(mask, 0).type_as(voxel_features)

        pillar_x = voxel_features[:, :, 0].unsqueeze(0).unsqueeze(0)
        pillar_y = voxel_features[:, :, 1].unsqueeze(0).unsqueeze(0)
        pillar_z = voxel_features[:, :, 2].unsqueeze(0).unsqueeze(0)
        pillar_i = voxel_features[:, :, 3].unsqueeze(0).unsqueeze(0)
        num_points = voxel_num_points.float().unsqueeze(0)

        pillarx = pillar_x.numpy()
        pillary = pillar_y.numpy()
        pillarz = pillar_z.numpy()
        pillari = pillar_i.numpy()
        numpoints = num_points.numpy()
        xsub_shaped = x_sub_shaped.numpy()
        ysub_shaped = y_sub_shaped.numpy()
        mask_np = mask.numpy()

        pillar_len = pillarx.shape[2]
        if pillar_len < 100:
            len_padding = 100 - pillar_len
            pillarx_pad = np.pad(pillarx, ((0,0),(0,0),(0,len_padding),(0,0)),'constant',constant_values=0)
            pillary_pad = np.pad(pillary, ((0,0),(0,0),(0,len_padding),(0,0)),'constant',constant_values=0)
            pillarz_pad = np.pad(pillarz, ((0,0),(0,0),(0,len_padding),(0,0)),'constant',constant_values=0)
            pillari_pad = np.pad(pillari, ((0,0),(0,0),(0,len_padding),(0,0)),'constant',constant_values=0)
            nump_pad = np.pad(numpoints, ((0,0),(0,len_padding)),'constant',constant_values=0)
            xsub_pad = np.pad(xsub_shaped, ((0,0),(0,0),(0,len_padding),(0,0)),'constant',constant_values=0)
            ysub_pad = np.pad(ysub_shaped, ((0,0),(0,0),(0,len_padding),(0,0)),'constant',constant_values=0)
            mask_pad = np.pad(mask_np, ((0,0),(0,0),(0,len_padding),(0,0)),'constant',constant_values=0)
        else:
            pillarx_pad = pillarx[:,:,:100,:]
            pillary_pad = pillary[:,:,:100,:]
            pillarz_pad = pillarz[:,:,:100,:]
            pillari_pad = pillari[:,:,:100,:]
            nump_pad = numpoints[:,:100]
            xsub_pad = xsub_shaped[:,:,:100,:]
            ysub_pad = ysub_shaped[:,:,:100,:]
            mask_pad = mask_np[:,:,:100,:]
        
        pillar_x_tensor =  torch.from_numpy(pillarx_pad)
        pillar_y_tensor =  torch.from_numpy(pillary_pad)
        pillar_z_tensor =  torch.from_numpy(pillarz_pad)
        pillar_i_tensor =  torch.from_numpy(pillari_pad)
        num_points_tensor =  torch.from_numpy(nump_pad)
        x_sub_shaped_tensor =  torch.from_numpy(xsub_pad)
        y_sub_shaped_tensor =  torch.from_numpy(ysub_pad)
        mask_tensor =  torch.from_numpy(mask_pad)

        #start_time = time.perf_counter()
        #net = self.net_pfe
        #ie = self.ie
        #input_blob = next(iter(net.input_info))
        #shape1 = net.input_info["pillar_x"].input_data.shape
        #net.reshape({'pillar_x':pillar_x.shape,
        #             'pillar_y':pillar_y.shape,
        #             'pillar_z':pillar_z.shape,
        #             'pillar_i':pillar_i.shape,
        #             'num_points_per_pillar':num_points.shape,
        #             'x_sub_shaped':x_sub_shaped.shape,
        #             'y_sub_shaped':y_sub_shaped.shape,
        #             'mask':mask.shape})
        #shape2 = net.input_info["pillar_x"].input_data.shape
        #exec_net_pfe = ie.load_network(network=net, device_name="CPU")
        #print('pfe_openvino load: %.2fms' %((time.perf_counter() - start_time)*1000))
        #print('pfe_openvino prepare: %.2fms' %((time.perf_counter() - start_time)*1000))
        inputs = {'pillar_x': pillar_x_tensor,
                                     'pillar_y': pillar_y_tensor,
                                     'pillar_z': pillar_z_tensor,
                                     'pillar_i': pillar_i_tensor,
                                     'num_points_per_pillar': num_points_tensor,
                                     'x_sub_shaped': x_sub_shaped_tensor,
                                     'y_sub_shaped': y_sub_shaped_tensor,
                                     'mask': mask_tensor}
        return inputs

    def forward(self, data_dict):
        raise NotImplementedError

    def sync_call(self, batch_dict):
        inputs_param = self.preprocessing(batch_dict)
        exec_net = self.exec_net_pfe
        #start_time = time.perf_counter()

        res = exec_net.infer(inputs=inputs_param)
        for k, v in res.items():
            if k == "174":
                res_torch = torch.as_tensor(v)
        voxel_features = res_torch.squeeze()
        voxel_features = voxel_features.permute(1, 0)
        batch_dict['pillar_features'] = voxel_features
        return batch_dict

    def async_call(self, batch_dict, inputs_param):
        if self.exec_net_pfe.requests[0].wait(0) == 0:
            self.exec_net_pfe.requests[0].wait(-1)

        self.frame_id = self.frame_id + 1
        request = self.exec_net_pfe.requests[0]
        request.set_completion_callback(py_callback=self.callback,
                                    py_data=(request, self.frame_id, batch_dict))
        self.event.clear()
        #print ("exec pfe net forward {}".format(self.frame_id))
        request.async_infer(inputs=inputs_param)
        return

    def postprocessing(self):
        self.event.wait()
        return self.queue.pop(0)

    def callback(self, statusCode, userdata):
        request, request_id, data_dict = userdata
        res = request.output_blobs
        for k, v in res.items():
            if k == "174":
                res_torch = torch.as_tensor(v.buffer)

        voxel_features = res_torch.squeeze()
        voxel_features = voxel_features.permute(1, 0)
        data_dict['pillar_features'] = voxel_features
        #voxelfeatures = voxel_features.numpy()
        #voxelfeatures.tofile("vfe_openPCD.bin")
        #print('pfe_openvino infer: %.2fms' %((time.perf_counter() - start_time)*1000))
        self.queue.append(data_dict)
        #print ("exec pfe net call back {}".format(request_id))
        self.event.set()

    def forward_vfe(self, batch_dict, **kwargs):
        start_time = time.perf_counter()

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
