import pickle
import time

import numpy as np
import torch
import tqdm
from pathlib import Path
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import cv2 
import torch
import csv
import os
import threading
import subprocess
from skimage import io
import numba

model_labels = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorbike",
    5: "bus",
    6: "train",
    7: "truck"
}

def yolo_launch(idx, input_node, output_node, datapath, save_to_file):
    num_labels = 5 #label_id, x_min, y_min, x_max, y_max
    input_node.write(str(idx))
    input_node.flush()
    print("yolo id {}".format(idx))
    data = os.read(output_node, 2048)
    #print('read data len {}'.format(len(data)))
    if len(data) == 0:
       print("nodata")
       time.sleep(0.1)

    bboxes = np.frombuffer(data, dtype=np.uint32).reshape(-1, num_labels)
    #print(bboxes)
    bboxes_dict = {
        'bboxes': bboxes[:, 1:],
        'classes': bboxes[:, :1],
    }
    if not save_to_file:
        return bboxes_dict, None

    img_file = datapath / 'image_2' / ('%s.png' % str(idx).zfill(6))
    assert img_file.exists()
    frame = cv2.imread(str(img_file))
    yolo_color = (0,255,0)

    for boxid, box in enumerate(bboxes):
        class_id = box[0]
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        det_label = model_labels[class_id]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), yolo_color, 2)
        cv2.putText(frame, 'yolo_{}'.format(boxid),
                    (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, yolo_color, 1)
        cv2.putText(frame, '{}'.format(det_label),
                    (xmin+67, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, yolo_color, 1)

    return bboxes_dict, frame

def pcl_launch(num, pcd, input_node, cond):
    print("Inside PCL Launch")
    cond.acquire()

    for i in range(num):
        val = cond.wait(10) #if wait 10s and there is no response from pcl, must be something wrong
        if val:
            pcd_file = Path(pcd) / ('%s.bin.pcd' % str(i).zfill(6))
            assert pcd_file.exists()
            #print ("pcl run on : %s" % str(pcd_file))
            input_node.write(str(pcd_file))
            input_node.flush()
            continue
        else:
            print("pcl wait timeout ...!")
            break
    cond.release()

def pcl_getresult(output_node):
    data = os.read(output_node, 2048)

    if len(data) == 0:
       print("nodata")
       time.sleep(0.1)
    pcl_pred = np.frombuffer(data, dtype=np.float32).reshape(-1, 6)#x_min, y_min, z_min, x_max, y_max, z_max

    #print("pcl_getresult, pcl raw data:")
    #print(pcl_pred)
    #end_time = int(round(time.time() * 1000))
    #print('\npcl_run2:', end_time - time1, 'ms')

    pred_pcd_box = [] #x, y, z, w, h, l, angle
    pred_pcd_boxes = []
    pred_pcd_scores = []
    pred_pcd_labels = []

    for idx, pcd in enumerate(pcl_pred):
        pred_pcd_box = [] #x, y, z, w, h, l, angle
        pred_pcd_box.append((pcd[0]+pcd[3])/2)
        pred_pcd_box.append((pcd[1]+pcd[4])/2)
        pred_pcd_box.append((pcd[2]+pcd[5])/2)
        pred_pcd_box.append(pcd[3]-pcd[0])
        pred_pcd_box.append(pcd[4]-pcd[1])
        pred_pcd_box.append(pcd[5]-pcd[2])
        pred_pcd_box.append(0) #fake angle result, todo, need calculated from 3D data

        pred_pcd_boxes.append(pred_pcd_box)
        pred_pcd_scores.append(0.5) #fake scores for pcl
        pred_pcd_labels.append(1) #fake labels for pcl, 'Car' - 1

    pred_pcd_boxes_tensor = torch.FloatTensor(pred_pcd_boxes)
    pred_pcd_scores_tensor = torch.FloatTensor(pred_pcd_scores)
    pred_pcd_labels_tensor = torch.tensor(pred_pcd_labels, dtype=torch.int)

    record_dict = {
        'pred_boxes': pred_pcd_boxes_tensor,
        'pred_scores': pred_pcd_scores_tensor,
        'pred_labels': pred_pcd_labels_tensor,
    }

    pred_dicts = []
    pred_dicts.append(record_dict)
    return pred_dicts       



def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

@numba.jit(nopython=True)
def _area(box):
    return max((box[2] - box[0]), 0) * max((box[3] - box[1]), 0)

@numba.jit(nopython=True)
def _iou(b1, b2, a1=None, a2=None):
    if a1 is None:
        a1 = _area(b1)
    if a2 is None:
        a2 = _area(b2)
    intersection = _area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                           min(b1[2], b2[2]), min(b1[3], b2[3])])

    u = a1 + a2 - intersection
    return intersection / u if u > 0 else 0



def fusion(frameid, bboxes_list, annos, output_path, datapath, save_to_file, corners3d):

    bboxes = bboxes_list['bboxes'] #2D bbox of camera
    classes = bboxes_list['classes']

    bboxes_lidar = annos[0]['bbox'] #2D bbox of lidar by projecting
    lboxes = annos[0]['boxes_lidar'] #3D bbox of lidar
    # print(f'lboxes 3D :: {lboxes}')
    cost_matrix = np.zeros((len(bboxes), len(bboxes_lidar)), dtype=np.float32)

    for i, idx in enumerate(bboxes):
        for j, d in enumerate(bboxes_lidar):
            iou_dist = _iou(idx, d)
            cost_matrix[i, j] = iou_dist

    cost = cost_matrix
    #print ("cost matric is: ")
    #print (cost)

    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
    #print ("row is: ")
    #print (row_ind)
    #print ("col is: ")
    #print (col_ind)

    #print ("sum of cost is: ")
    #print (cost[row_ind, col_ind].sum())
    #print ("assignment cost is: ")
    #print (cost[row_ind, col_ind])

    threshold = 0.1
    selected = []
    selected_lidar = []
    cost_confidence = []
    for i in range(len(row_ind)):
        if cost[row_ind[i], col_ind[i]] > threshold:
            selected.append(i)
            selected_lidar.append(i)
            cost_confidence.append(cost[row_ind[i], col_ind[i]])
    seleted_bboxes = bboxes[selected]
    selected_lidar_bbox = corners3d[selected_lidar]
    # print(f'selected box :: {seleted_bboxes}')

    if not save_to_file: 
        return None

    img_file = datapath / 'image_2' / ('%s.png' % str(frameid).zfill(6))
    assert img_file.exists()    
    cur_img_file = output_path / ('fuse_%s.png' % str(frameid).zfill(6))
    cur_3dbb_file = output_path / ('3dbb_%s.png' % str(frameid).zfill(6))
    frame = cv2.imread(str(img_file))

    #BGR
    lidar_color = (255,255,255)
    yolo_color = (0,255,0)
    #fuse_color = (200,0,200)

    for idx in range(len(selected_lidar_bbox)):
    # for idx in range(len(seleted_bboxes)):
        xmin = int(seleted_bboxes[idx][0])
        ymin = int(seleted_bboxes[idx][1])
        xmax = int(seleted_bboxes[idx][2])
        ymax = int(seleted_bboxes[idx][3])

        x_color = int(((lboxes[col_ind[selected[idx]]][0] if lboxes[col_ind[selected[idx]]][0] <= 70 else 70)/70) * 255)
        lidar_color = (255, x_color, x_color)
        classid = classes[row_ind[selected[idx]]]
        det_label = model_labels[classid[0]]
        label_confidence = det_label + '_' + str(round(cost_confidence[idx],2))

        if(cost_confidence[idx] > 0.2):
            cv2.putText(frame, 'lidar_{}'.format(label_confidence),
                    (xmin+67, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, yolo_color, 1)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), lidar_color, 2)
            # cv2.putText(frame, 'lidar_{}'.format(row_ind[selected[idx]]), (xmin, ymin - 7),
            #         cv2.FONT_HERSHEY_COMPLEX, 0.6, yolo_color, 1)
            # cv2.putText(frame, 'lidar_{}'.format(col_ind[selected[idx]]), (xmin, ymin + 17),
            #         cv2.FONT_HERSHEY_COMPLEX, 0.6, lidar_color, 1)
            frame = draw_projected_box3d(frame, selected_lidar_bbox[idx])


    # cv2.imwrite(str(cur_img_file), frame)

    # cv2.imwrite(str(cur_3dbb_file), frame)

    print ("Save fusion result to: %s" % cur_img_file)
    return frame

def draw_projected_box3d(image, qs, color=(200,0,200), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    # print(f'qs shape :: {qs.shape}')
    # print(f'QS :: {qs}')
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image

def yolo_start(cond, num, filepath):
    cond.acquire()
    print(" I am here : Object detection demo")
    yolo_p = subprocess.Popen(["./object_detection_demo", "-at", "yolo",
                                "-i", filepath,
                                "-m", "./yolo_v3.xml",
                                "-d", "CPU", "-no_show",
                                "-num", str(num),
                                "-nireq", "1"]) #vscode need update path
    print("-----------------yolo server start\n")
    cond.wait()
    yolo_p.terminate()
    print("-----------------yolo server terminate")

def pcl_start(cond, num, filepath):
    #start pcl server, communicate with it by pipe file
    cond.acquire()
    print(f" I am here : PCL Obj detection : {num}")
    pcl_p = subprocess.Popen(["./pcl_object_detection", "-num", str(num), "-pcd", filepath]) #vscode need update path
    print("-----------------pcl server start\n")
    cond.wait()
    pcl_p.terminate()   
    print("-----------------pcl server terminate")

def eval_one_epoch(num, cfg,
    model, dataloader, epoch_id, logger,
    dist_test=False, save_to_file=False, result_dir=None):

    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = Path(__file__).resolve().parents[1] / 'output'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    datapath = dataset.get_datapath()
    det_annos = []
    img_set = []

    print('Reading images needs some time...\n')
    # for dataid in range(num+1 if num else len(dataset)+1):
    for dataid in range(num if num else len(dataset)):
        print(f'Data id: {dataid}')
        img_file = datapath / 'image_2' / ('%s.png' % str(dataid).zfill(6))
        cam_img = io.imread(img_file)
        img_set.append(cam_img)
    print('There are {} images in total.\n'.format(dataid))
    dataset.set_dataptr(img_set)

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)

    pcd_path = datapath / 'velodyne_reduced_pcd'
    if not os.path.exists(pcd_path):
        print("{} : does not exist".format(pcd_path))
        return {}
    file_num = num if num else len(dataset)
    cond_sv = threading.Condition()
    pcl_server = threading.Thread(target=pcl_start, args=(cond_sv, file_num, pcd_path,))
    pcl_server.start()

    img_path = datapath / 'image_2'
    if not os.path.exists(img_path):
        print("{} : does not exist".format(pcd_path))
        return {}
    cond_sv_yolo = threading.Condition()
    yolo_server = threading.Thread(target=yolo_start, args=(cond_sv_yolo, file_num, img_path,))
    yolo_server.start()

    FILE_PATH = str(Path(__file__).resolve().parents[2]/'tools'/'file.fifo') #vscode need update the path
    OBJECT_DATA = str(Path(__file__).resolve().parents[2]/'tools'/'data.fifo')
    print("File path " + FILE_PATH)
    print("Object data " + OBJECT_DATA)
    # Comment
    FILE_PATH_YOLO = str(Path(__file__).resolve().parents[2]/'tools'/'file_yolo.fifo')
    OBJECT_DATA_YOLO = str(Path(__file__).resolve().parents[2]/'tools'/'data_yolo.fifo')
    print("File path Yolo " + FILE_PATH_YOLO)
    print("Object data Yolo " + OBJECT_DATA_YOLO)
    
    print("Sleeping 2 secs")
    time.sleep(2)

    if not os.path.exists(FILE_PATH):
        os.mkfifo(FILE_PATH, 0o666)
    if not os.path.exists(OBJECT_DATA):
        os.mkfifo(OBJECT_DATA, 0o666)
    
    pcl_input = open(FILE_PATH, 'w') #TODO, why open and why os.open?
    pcl_output = os.open(OBJECT_DATA, os.O_RDONLY)

    # Comment
    print("Sleeping 2 secs")
    time.sleep(2)

    if not os.path.exists(FILE_PATH_YOLO):
        os.mkfifo(FILE_PATH_YOLO, 0o666)
    if not os.path.exists(OBJECT_DATA_YOLO):
        os.mkfifo(OBJECT_DATA_YOLO, 0o666)

    yolo_input = open(FILE_PATH_YOLO, 'w')
    yolo_output = os.open(OBJECT_DATA_YOLO, os.O_RDONLY)

    cond = threading.Condition()
    print('Going to launch PCL launch thread')
    pcl_thread = threading.Thread(target=pcl_launch,
            args=(num if num else len(dataset), pcd_path, pcl_input, cond,))
    pcl_thread.start()

    start_time = time.time()
    #yolo_time = 0
    #pcl_time = 0
    #fuse_time = 0
    #pre_time = 0
    #pre_start_time = time.time()
    start_count = 1 #start time counting from start_count+1 frame

    for i, batch_dict in enumerate(dataloader):
        if num > 0: 
            if i >= num:
                break
        #if i > start_count:
        #    pre_time += time.time() - pre_start_time
        #pcl_start_time = time.time()
        cond.acquire()
        cond.notify()
        cond.release()

        #yolo_start_time = time.time()
        # Comment
        bboxes_list, img_cam = yolo_launch(i, yolo_input, yolo_output, datapath, save_to_file)
        # yolo_time += time.time() - yolo_start_time
        # obj_detect.frameid = i
        # bboxes_list, img_cam = obj_detect.obj_detect_run()

        load_data_to_gpu(batch_dict)
        batch_dict['frameid'] = i

        if 1:
            pred_dicts = pcl_getresult(pcl_output)
            #pcl_time += time.time() - pcl_start_time
        else:
            with torch.no_grad():
                pred_dicts = model.latency(batch_dict) #not test this path with pointpillars

        disp_dict = {}
        # statistics_info(cfg, ret_dict, metric, disp_dict) #TODO, need to add recall rate without CUDA
        annos, img_lidar, corner_box = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None)

        # print(f'3d bbox corners3d :: {corners3d}')
        #fuse_start_time = time.time()
        # Comment
        if bboxes_list:
            img_fuse = fusion(i, bboxes_list, annos, final_output_dir, datapath, save_to_file, corner_box)
        #fuse_time += time.time()-fuse_start_time
        
        # Comment
        if save_to_file:
            yolo_file = final_output_dir / ('yolo_%s.png' % str(i).zfill(6))
            # cv2.imwrite(str(yolo_file), img_cam)
            vis = np.concatenate((img_cam, img_lidar, img_fuse), axis=0)
            out_file = final_output_dir / ('merge_%s.png' % str(i).zfill(6))
            # cv2.imwrite(str(out_file), vis)
            print ("Save merge result to: %s" % out_file)

        det_annos += annos
        #if cfg.LOCAL_RANK == 0:
        #    progress_bar.set_postfix(disp_dict)
        #    progress_bar.update()
        #pre_start_time = time.time()
        if i == start_count:
            start_time = time.time()

    end_time = time.time()

    cond_sv.acquire()
    cond_sv.notify()
    cond_sv.release()
    time.sleep(1)
    pcl_input.close()
    os.close(pcl_output)

    cond_sv_yolo.acquire()
    cond_sv_yolo.notify()
    cond_sv_yolo.release()
    time.sleep(1)
    # Comment
    yolo_input.close()
    os.close(yolo_output)

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    #logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    #logger.info('*************** len %s *****************' % len(dataset))
    #sec_per_example = (time.time() - start_time) / len(dataset)
    #logger.info('Generate label finished(sec_per_example: %.2f ms).\n' % sec_per_example*1000)
    print("\ntotal {} frame takes {:.2f}ms, {:.2f}ms per frame, fps = {:.2f}\n\n".format(
                i-start_count-1, (end_time-start_time)*1000,
                (end_time-start_time)*1000/(i-start_count-1),
                (i-start_count-1)/(end_time-start_time)))
    #print("pcl\t\t takes avg {:.2f}ms\t on each frame".format(pcl_time*1000/i))
    #print("yolo\t\t takes avg {:.2f}ms\t on each frame".format(yolo_time*1000/i))
    #print("fuse\t\t takes avg {:.2f}ms\t on each frame".format(fuse_time*1000/i))
    #print("preprocess\t takes avg {:.2f}ms\t on each frame\n".format(pre_time*1000/(i-start_count-1)))

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        #logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        #logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
               % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    # """
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)
    logger.info(ret_dict)
    # """
    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
