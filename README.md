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

![Error](https://user-images.githubusercontent.com/16526627/211657957-0d23989a-dceb-4756-9645-d9df5beee611.png)

## OpenVINO based Implementation for Intel NCS2 Stick

- After reducing the number of pointpillar features (voxels), we were able to fit the IR (intermediate representation) model into the Intel NCS2 Device! Although this meant a performance issue which we will see next.
- We used the github link from intels repository as [reference](https://github.com/intel/OpenVINO-optimization-for-PointPillars) for the implementation.
- We prepared the dataset as mentioned in the [Smallmunich](https://github.com/SmallMunich/nutonomy_pointpillars#onnx-ir-generate) repo to create the reduced version of the dataset from kitti dataset.
- Code modified to optimize for Intel NCS2 Myriad stick as the code supports only intel CPUs and GPUs.
- We will be using pointpillars pretrained model from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet#kitti-3d-object-detection-baselines) repository.
- The model configuration can be seen in the pointpillar.yaml file.
- As you can see from the file, the NMS threshold for 3D bounding box is set to 0.1 and learning rate is set to 0.003 with Adam optimizer.
- Train test split of 50/50 is used for the model.

## Sample Output

![Sample Output](https://user-images.githubusercontent.com/16526627/211658888-2ef39636-3e38-49a5-99fa-dc45b35936fd.png)

## Steps to load and run models on Intel NCS2

Step 1: Install OpenVINO toolkit for linux as shown [here](https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_linux.html).

Step 2: Configure NCS2 stick USB rules as shown [here](https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_linux.html#optional-steps-for-intel-neural-compute-stick-2).

Step 3: If you have an existing model, you can use the Model Optimizer to convert the model to intermediate Representation (IR) to run on NCS.

Step 4: Output of the MO will be .xml and .bin files which contain the neural network weights and architecture.

Step 5: Use ‘lsusb’ command to check if the MYRIAD device is seen.

Step 6: We will use the OpenPCDet codebase to perform the [MO](https://github.com/pointpillars-on-openvino/pointpillars-on-openvino/blob/main/0001-implement-pointpillars-on-Intel-platform.patch) for OpenVINO.

## Model Performance

### 1 Edge (NCS2 Stick) device

FPS : 0.32

![pcl_1_ncs_framedrop](https://user-images.githubusercontent.com/16526627/211661189-7ce74c3c-17e4-4ffc-8a9b-44a5c0560116.gif)

### 2 Edge (NCS2 Stick) devices

FPS : 0.66

![pcl_2_ncs_framedrop](https://user-images.githubusercontent.com/16526627/211661259-60fa55ce-76c9-4fa4-9d1f-7c1f9142b77a.gif)



