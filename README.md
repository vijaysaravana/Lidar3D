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

![Architecture Diagram](https://user-images.githubusercontent.com/16526627/211655233-ca888f6f-87a4-483b-b686-224d3227689e.png)





