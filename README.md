# AbnormalBehaviorDetection

## Requirements
* Ubuntu 22.04
* CUDA 11.3 <=
* Docker.ce
* Docker-compose

## Installation
```shell
git clone --recursive https://github/jinhasong/AbnormalBehaviorDetection.git
cd AbnormalBehaviorDetection
docker-compose up -d
```
## Training
```shell

```
## Evaluation
```shell

```
## Inference
```shell

```

## Project Structure

<details><summary> <b>Expand</b> </summary>

``` text
AbnormalBehaviorDetection/
 ├─ docker/
 │   ├─ dev/
 │   ├─ mmaction2/
 │   ├─ pose/
 │   ├─ yolo/
 ├─ lib
 │   ├─ mmaction2/
 │   ├─ pose/
 │   │   └─ alphapose/  
 │   └─ tracker/
 │       ├─ bytetracker/
 │       └─ sort/
 ├─ models/
 ├─ scripts/
 ├─ tools/
 ├─ utils/
 └─ docker-compose.yml
```

- docker: docker 빌드를 위한 dockerfile, env 및 requirements.txt 파일 포함
- lib: git submodule로 추가한 관련 라이브러리 및 repository
- models: abnormal behavior detection 모델 class
- scripts: 데이터셋 및 모델 다운로드용 shell scripts
- tools: 테스트용 소스 및 툴
- utils: model, tools, lib 외 utility

</details>

## Note
<details><summary> <b>Expand</b> </summary>

### Object Detection([model comparison](https://jinhasong.github.io/AbnormalBehaviorDetection/yolo.html))
|  Name  | Paper                                                                                                                          |                   Github                   | 
|:------:|:-------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------:|
| yolov4 | [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)                                     |    https://github.com/AlexeyAB/darknet     |
| yolov7 | [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696) |    https://github.com/WongKinYiu/yolov7    |
| yolov8 | [YOLOv8 Docs](https://docs.ultralytics.com/)                                                                                   | https://github.com/ultralytics/ultralytics |
| yolov9 | [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)            |    https://github.com/WongKinYiu/yolov9    |

### Object Tracking
|    Name    | Paper                                                                                                   |                  Github                   | 
|:----------:|:--------------------------------------------------------------------------------------------------------|:-----------------------------------------:|
| ByteTrack  | [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864) |   https://github.com/ifzhang/ByteTrack    |
|    SORT    | [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)                                 |      https://github.com/abewley/sort      |

### Pose Estimation
|   Name    | Paper                                                                                                                     |                 Github                 | 
|:---------:|:--------------------------------------------------------------------------------------------------------------------------|:--------------------------------------:|
| Alphapose | [AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time](https://arxiv.org/abs/2211.03375) | https://github.com/MVIG-SJTU/AlphaPose |
|  MMPose   | [Papers Link](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html)                                   |  https://github.com/open-mmlab/mmpose  |

### Action Recognition
|  Name   | Paper                                                                            |                                        Github                                         | 
|:-------:|:---------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------:|
| PoseC3D | [Revisiting Skeleton-based Action Recognition](https://arxiv.org/abs/2104.13586) | https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/README.md  |
- [MMAction2](https://github.com/open-mmlab/mmaction2)

### Abnormal Event Detection
#### Dataset
##### [Violence(Assault, Fight)](https://paperswithcode.com/sota/abnormal-event-detection-in-video-on-ubi)
| Type  |     Name     | Metric |                                      Link                                       |                 Dataset Path                 | Downloaded |
|:-----:|:------------:|:------:|:-------------------------------------------------------------------------------:|:--------------------------------------------:|:----------:|
| Eval  | CUHK-Avenue  |  AUC   | [link](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) | ```mlsun/nfs_shared/abd/eval/CUHK-Avenue```  |    [x]     |
| Eval  | ShanghaiTech |  AUC   |            [link](https://github.com/desenzhou/ShanghaiTechDataset)             | ```mlsun/nfs_shared/abd/eval/ShanghaiTech``` |    [x]     |
| Eval  |   UBnormal   |  AUC   |               [link](https://github.com/lilygeorgescu/UBnormal/)                |   ```mlsun/nfs_shared/abd/eval/UBnormal```   |    [x]     |
| Eval  |  UCF-Crime   |  AUC   |              [link](https://paperswithcode.com/dataset/ucf-crime)               |  ```mlsun/nfs_shared/abd/eval/UCF_Crimes```  |    [x]     |
| Train |   LAD2000    |        |         [link](https://github.com/wanboyang/anomaly_detection_LAD2000)          |      ```mlsun/nfs_shared/abd/LAD2000```      |    [x]     |
| Train | NWPU-Campus  |        |                      [link](https://campusvad.github.io/)                       |    ```mlsun/nfs_shared/abd/NWPU_Campus```    |    [x]     |
| Train | StreetScene  |        |    [link](https://www.merl.com/research/highlights/video-anomaly-detection)     |    ```mlsun/nfs_shared/abd/StreetScene```    |    [ ]     |
| Eval  | XD-Violence  |   AP   |             [link](https://paperswithcode.com/dataset/xd-violence)              |   ```mldisk2/nfs_shared/abd/XD-Violence```   |    [x]     |


##### Fall
| Type  |     Name     | Metric |                                      Link                                       |                 Dataset Path                 | Downloaded |
|:-----:|:------------:|:------:|:-------------------------------------------------------------------------------:|:--------------------------------------------:|:----------:|

##### Wander
| Type  |     Name     | Metric |                                      Link                                       |                 Dataset Path                 | Downloaded |
|:-----:|:------------:|:------:|:-------------------------------------------------------------------------------:|:--------------------------------------------:|:----------:|

#### References
##### Violence(Assault, Fight)
| Method |                                                                        Paper                                                                         |                         Github                          | Framework | CUHK-Avenue(ROC) | ShanghaiTech | UCF-Crime | UBnormal |
|:------:|:----------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------:|:---------:|:----------------:|:------------:|:---------:|:--------:|
|  BODS  |                                                       [Link](https://arxiv.org/abs/1908.05884)                                                       |                            -                            |    AE     |        -         |      -       |   68.2    |    -     |
|  GODS  |                                                       [Link](https://arxiv.org/abs/1908.05884)                                                       |                            -                            |    AE     |        -         |      -       |   69.4    |    -     |
|  VEC   |                                                       [Link](https://arxiv.org/abs/2008.11988)                                                       |         https://github.com/yuguangnudt/VEC_VAD          |  AE(FPM)  |       90.2       |     74.8     |     -     |    -     |
|  CAC   |                                              [Link](https://dl.acm.org/doi/abs/10.1145/3394171.3413529)                                              |                            -                            |  Encoder  |       87.0       |     79.3     |     -     |    -     |
|  HF2   |                                                       [Link](http://arxiv.org/abs/2108.06852)                                                        |           https://github.com/LiUzHiAn/hf2vad            |  AE(FPM)  |       91.1       |     76.2     |     -     |    -     |
|  BAF   |                                            [Link](https://ieeexplore.ieee.org/abstract/document/9410375)                                             |          https://github.com/lilygeorgescu/AED           |    AE     |       92.3       |     82.7     |     -     |   59.3   |
|  BDPN  |                                            [Link](https://ojs.aaai.org/index.php/AAAI/article/view/19898)                                            |          https://github.com/lilygeorgescu/AED           |  AE(FPM)  |       90.0       |     78.1     |     -     |    -     |
|  GCL   |                                                       [Link](http://arxiv.org/abs/2203.03962)                                                        |                            -                            |    GAN    |        -         |     79.6     |   74.2    |    -     |
|  SSL   |                                                       [Link](https://arxiv.org/pdf/2207.10172)                                                       |         https://github.com/gdwang08/Jigsaw-VAD          |  Encoder  |       92.2       |     84.3     |     -     |    -     |
|        |                                            [Link](https://ieeexplore.ieee.org/abstract/document/10222594)                                            | https://github.com/AnilOsmanTur/video_anomaly_diffusion |           |        -         |     76.1     |   65.2    |    -     |
|  FPDM  | [Link](https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.pdf) |           https://github.com/daidaidouer/FPDM           |   DDIM    |       90.1       |     78.6     |   74.7    |   62.7   |

### OpticalFlow
- [Awesome-Optical-Flow](https://github.com/hzwer/Awesome-Optical-Flow?tab=readme-ov-file)

</details>
