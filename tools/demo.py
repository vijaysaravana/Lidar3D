import argparse
import glob
from logging import Logger
from pathlib import Path
import threading

import numpy as np
import torch
import time
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

buffer = []
function_buffer = []

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin', ram=False):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        assert data_file_list, 'Make sure there are point data (.bin or .npy) in the folder.'

        data_file_list.sort()
        self.sample_file_list = data_file_list
        self.points_set = []
        if ram:
            for idx in range(len(self.sample_file_list)):
                if self.ext == '.bin':
                    points = np.fromfile(self.sample_file_list[idx], dtype=np.float32).reshape(-1, 4)
                elif self.ext == '.npy':
                    points = np.load(self.sample_file_list[idx])
                else:
                    raise NotImplementedError
                self.points_set.append(points)
    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.points_set:
            points = self.points_set[index]
        elif self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='pointpillar.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/home/iotg/work/kitti_dataset/training/velodyne_reduced',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--mode', type=str, default='balance', help='specify the pineline excute mode: balance throughput latency or all')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--num', type=int, default='-1', help='specify how many files are used. -1 mean all')
    parser.add_argument('--debug', action='store_true', default=False, help='specify if enable debug result')
    parser.add_argument('--ram', action='store_true', default=False, help='specify if read the dataset firstly')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def run(model, mode, num, demo_dataset, logger, debug_print):
    logger.info('------run number of samples: \t{} in mode: {}'.format(num, mode))
    func_dict = {
                'balance': model.balance,
                'throughput': model.throughput,
                'latency': model.latency,
                };

    exec_fun = func_dict[mode]

    device = 'cpu'
    model.to(device)

    model.eval()
    start_time_2 = time.perf_counter()
    frames = 0
    start_time = []
    end_time = []
    latency = []

    with torch.no_grad():
        start_time.append(time.perf_counter())
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            data_dict['frameid'] = idx
            # print("Data dict")
            # print(data_dict)
            load_data_to_gpu(data_dict)
            # Add this to queue
            buffer.append(data_dict)
            function_buffer.append(exec_fun)
            # print(f'Added data dict {idx} to buffer')
            pred_dicts = exec_fun(data_dict)
            logger.info (pred_dicts) if debug_print and pred_dicts else None
            start_time.append(time.perf_counter())
            frames = frames + 1
            if frames >= num:
                break
        pred_dicts = exec_fun(None)
        logger.info (pred_dicts) if debug_print and pred_dicts else None
        start_time.pop()

    total_time = time.perf_counter() - start_time_2
    end_time = model.end_time
    latency = np.array(end_time) - np.array(start_time)
    if len(latency) > 20:
        latency = latency[10:-10]  ##remove the first and last 10 elements
    elif len(latency) > 4:
        latency = latency[1:-1]  ##remove the first and last 1 elements

    if total_time < 1:
        logger.info('total: \t\t%.2f milliseconds' % (total_time*1000))
    else:
        logger.info('total: \t\t%.2f seconds' % (total_time))
    logger.info('FPS: \t\t%.2f' % (frames/total_time))
    logger.info('latency: \t%.2f milliseconds' % (np.mean(latency) * 1000))

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()

    modes = ['balance', 'throughput', 'latency']
    run_mode = []
    if args.mode == 'all':
        run_mode = modes
    elif args.mode in modes:
        run_mode.append(args.mode)
    else:
        logger.error ("The mode is not support, please select in {}".format(list(modes)))
        logger.error ("Or could set mode to 'all' to run all the modes above")
        exit(1)

    num = args.num
    if num != -1 and num < 5 and args.mode != 'latency':
        logger.error ("at least 5 frames are need for mode balance and throughput.")
        exit(1)

    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    logger.info('Loading the dataset and model.')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger, ram=args.ram
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)

    num = len(demo_dataset) if num == -1 else num

    logger.info('number of samples in dataset: \t{}'.format(len(demo_dataset)))
    for mode in run_mode:
        run(model, mode, num, demo_dataset, logger, args.debug)

    logger.info('Demo done.')

def predict_async_ncs(thread_id):
    try: 
        # logger = common_utils.create_logger()
        if(len(buffer) > 0 and len(function_buffer) > 0):
            exec_fun = function_buffer.pop()
            data_dict = buffer.pop()
            print(f'Received {data_dict} from buffer {thread_id}')
            pred_dicts = exec_fun(data_dict)
            # logger.info (pred_dicts) if pred_dicts else None
    except:
        import traceback
        traceback.print_exc()

def async_infer_ncs(thread_id):

    #ncsworker.skip_frame_measurement()

    while True:
        predict_async_ncs(thread_id)


if __name__ == '__main__':
    
    # threads = []
    # for devid in range(2):
    #     thworker = threading.Thread(target=async_infer_ncs, args=(devid,))
    #     thworker.start()
    #     threads.append(thworker)
    main()
    # print("Created threads")
    # for th in threads:
    #     th.join()
    
    

