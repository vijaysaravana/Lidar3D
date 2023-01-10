import numpy as np
import torch
import torch.nn as nn
import threading
from pathlib import Path

#BaseBEVBackbone_ASYNC=False
BaseBEVBackbone_ASYNC=True

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, openvino_ie):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        self.frame_id = 0
        self.event = threading.Event()
        self.queue = []

        self.ie = openvino_ie
        model_file_rpn = str(Path(__file__).resolve().parents[3] / 'tools'/'rpn.xml')
        model_weight_rpn = str(Path(__file__).resolve().parents[3] / 'tools'/'rpn.bin')
        self.net_rpn = openvino_ie.read_network(model_file_rpn, model_weight_rpn)
        self.exec_net_rpn = openvino_ie.load_network(network=self.net_rpn, device_name="CPU")

        myriad1_config = {}
        myriad2_config = {}

        # self.ie.set_config(config=myriad1_config, device_name="MYRIAD.1.1.4-ma2480")
        # self.ie.set_config(config=myriad2_config, device_name="MYRIAD.1.1.1-ma2480")

        # self.exec_net_rpn = self.ie.load_network(
        #     network=self.net_rpn, device_name="MULTI", num_requests=8, config={"MULTI_DEVICE_PRIORITIES": "MYRIAD.1.1.4-ma2480,MYRIAD.1.1.1-ma2480"}
        # )

        # Query the optimal number of requests
        nireq = self.exec_net_rpn.get_metric("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        print(f'Optimal number of Infer req RPN :: {nireq}')

    def forward_backbone2d(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

    def callback(self, statusCode, userdata):
        request, request_id, data_dict = userdata
        #print ("exec rpn net call back {}".format(request_id))
        res = request.output_blobs
        for k, v in res.items():
            if k == "184":
                data_dict['batch_box_preds'] = torch.as_tensor(v.buffer)
            elif k == "185":
                data_dict['batch_cls_preds'] = torch.as_tensor(v.buffer)
            elif k == "187":
                data_dict['dir_cls_preds'] = torch.as_tensor(v.buffer)
        self.queue.append(data_dict)
        self.event.set()

    def forward(self, data_dict):
        raise NotImplementedError

    def preprocessing(self, data_dict, **kwargs):
        input_blob = next(iter(self.exec_net_rpn.input_info))
        return {input_blob: data_dict['spatial_features']}

    def sync_call(self, data_dict):
        #start_time = time.perf_counter()
        inputs_param = self.preprocessing(data_dict)
        request = self.exec_net_rpn.requests[0]
        res = self.exec_net_rpn.infer(inputs=inputs_param)
        for k, v in res.items():
            if k == "184":
                data_dict['batch_box_preds'] = torch.as_tensor(v)
            elif k == "185":
                data_dict['batch_cls_preds'] = torch.as_tensor(v)
            elif k == "187":
                data_dict['dir_cls_preds'] = torch.as_tensor(v)
        return data_dict

    def postprocessing(self):
        self.event.wait()
        return self.queue.pop(0)

    def async_call(self, batch_dict, inputs_param):
        if self.exec_net_rpn.requests[0].wait(0) == 0:
                self.exec_net_rpn.requests[0].wait(-1)

        self.frame_id = self.frame_id + 1
        request = self.exec_net_rpn.requests[0]
        request.set_completion_callback(py_callback=self.callback,
                                    py_data=(request, self.frame_id, batch_dict))
        self.event.clear()
        #print ("exec rpn net forward back {}".format(self.frame_id))
        request.async_infer(inputs=inputs_param)
        return

