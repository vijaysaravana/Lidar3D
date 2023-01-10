# Lidar based 3D Object detection using OpenPCDet on Intel NCS2 Devices with OpenVINO framework

This repo contains the code based on Intel's OpenVINO Optimization for Lidar Based 3D object detection to be run on Intel NCS2 Devices. The repo is based on the intel implementation [here](https://github.com/intel/OpenVINO-optimization-for-PointPillars). 


## Steps to run

Navigate to tools folder and run the test.py file as follows : 

```
python test.py --cfg_file pointpillar.yaml
```

## OpenPCDet

`OpenPCDet` is a clear, simple, self-contained open source project for LiDAR-based 3D object detection. 

It is also the official code release of [`[PointRCNN]`](https://arxiv.org/abs/1812.04244), [`[Part-A^2 net]`](https://arxiv.org/abs/1907.03670) and [`[PV-RCNN]`](https://arxiv.org/abs/1912.13192). 

## Model Inference Pipeline

Based on the Camera / Lidar Data from the KiTTi Dataset, we will preprocess the data using the YoLo Object detection module and Point Cloud Library (PCL) object detection module correspondingly in parallel. The PCL Object detection will occur on NCS2 Intel Stick and 3D bounding boxes will be constructed. Based on the overlapping threshold of 3D PCL bounding box and 2D YoLo bounding box final results are produced.

![Architecture Diagram](https://user-images.githubusercontent.com/16526627/211656670-5f874e74-d5fc-4e25-ba08-84285bb45f2e.png)

## Errors faced while running on NCS2

- We faced `OUT_OF_MEMORY` error while trying to load the model into Intel NCS 2.
- Intel NCS 2 has a memory size of 2.5 MB. The point pillars model was large and this could not fit in the NCS device.
- Decreased the number of stacked Pointpillars voxels gradually from 12000 to 100 in steps.
