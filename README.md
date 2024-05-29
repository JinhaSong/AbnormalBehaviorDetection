# AbnormalBehaviorDetection

## Requirements
* Ubuntu 22.04
* CUDA 11.3 <=
* Docker.ce
* Docker-compose

## Dataset Information
* [docs/dataset.md](docs/dataset.md)

## Installation
```shell
git clone --recursive https://github/jinhasong/AbnormalBehaviorDetection.git
cd AbnormalBehaviorDetection
docker-compose up -d
```

## Preprocessing
```shell

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
|  Name  | Paper                                                                                                                          |                          Github                           | 
|:------:|:-------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------:|
| yolov4 | [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)                                     |    [Github Link](https://github.com/AlexeyAB/darknet)     |
| yolov7 | [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696) |    [Github Link](https://github.com/WongKinYiu/yolov7)    |
| yolov8 | [YOLOv8 Docs](https://docs.ultralytics.com/)                                                                                   | [Github Link](https://github.com/ultralytics/ultralytics) |
| yolov9 | [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)            |    [Github Link](https://github.com/WongKinYiu/yolov9)    |

### Object Tracking
|    Name    | Paper                                                                                                   |                       Github                        | 
|:----------:|:--------------------------------------------------------------------------------------------------------|:---------------------------------------------------:|
| ByteTrack  | [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864) | [Github Link](https://github.com/ifzhang/ByteTrack) |
|    SORT    | [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)                                 |   [Github Link](https://github.com/abewley/sort)    |

### Pose Estimation
|   Name    | Paper                                                                                                                     |                        Github                         | 
|:---------:|:--------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------:|
| Alphapose | [AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time](https://arxiv.org/abs/2211.03375) | [Github Link](https://github.com/MVIG-SJTU/AlphaPose) |
|  MMPose   | [Papers Link](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html)                                   |  [Github Link](https://github.com/open-mmlab/mmpose)  |

### Action Recognition
|  Name   | Paper                                                                            |                                               Github                                                | 
|:-------:|:---------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------:|
| PoseC3D | [Revisiting Skeleton-based Action Recognition](https://arxiv.org/abs/2104.13586) | [Github Link](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/README.md) |
- [MMAction2](https://github.com/open-mmlab/mmaction2)

### Abnormal Event Detection
#### Dataset
##### [Video Anomaly Detection(VAD)](https://paperswithcode.com/sota/abnormal-event-detection-in-video-on-ubi)
| Type  |     Name     | Metric |                                      Link                                       |                 Dataset Path                 |       Downloaded        |
|:-----:|:------------:|:------:|:-------------------------------------------------------------------------------:|:--------------------------------------------:|:-----------------------:|
| Eval  | CUHK-Avenue  |  AUC   | [Link](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) | ```mlsun/nfs_shared/abd/eval/CUHK-Avenue```  | <ul><li>[x] </li></ul>  |
| Eval  | ShanghaiTech |  AUC   |            [Link](https://github.com/desenzhou/ShanghaiTechDataset)             | ```mlsun/nfs_shared/abd/eval/ShanghaiTech``` | <ul><li>[x] </li></ul>  |
| Eval  |   UBnormal   |  AUC   |               [Link](https://github.com/lilygeorgescu/UBnormal/)                |   ```mlsun/nfs_shared/abd/eval/UBnormal```   | <ul><li>[x] </li></ul>  |
| Eval  |  UCF-Crime   |  AUC   |              [Link](https://paperswithcode.com/dataset/ucf-crime)               |  ```mlsun/nfs_shared/abd/eval/UCF_Crimes```  | <ul><li>[x] </li></ul>  |
| Train |   LAD2000    |        |         [Link](https://github.com/wanboyang/anomaly_detection_LAD2000)          |      ```mlsun/nfs_shared/abd/LAD2000```      | <ul><li>[x] </li></ul>  |
| Train | NWPU-Campus  |        |                      [Link](https://campusvad.github.io/)                       |    ```mlsun/nfs_shared/abd/NWPU_Campus```    | <ul><li>[x] </li></ul>  |
| Train | StreetScene  |        |    [Link](https://www.merl.com/research/highlights/video-anomaly-detection)     |    ```mlsun/nfs_shared/abd/StreetScene```    | <ul><li>[ ] </li></ul>  |
| Eval  | XD-Violence  |   AP   |             [Link](https://paperswithcode.com/dataset/xd-violence)              |   ```mldisk2/nfs_shared/abd/XD-Violence```   | <ul><li>[x] </li></ul>  |


##### Fall
| Type  |     Name     | Metric |                                      Link                                       |                 Dataset Path                 | Downloaded |
|:-----:|:------------:|:------:|:-------------------------------------------------------------------------------:|:--------------------------------------------:|:----------:|

##### Wander
| Type  |     Name     | Metric |                                      Link                                       |                 Dataset Path                 | Downloaded |
|:-----:|:------------:|:------:|:-------------------------------------------------------------------------------:|:--------------------------------------------:|:----------:|

#### References

| Method |                                                                        Paper                                                                         |                                 Github                                 | Feature | Framework | CUHK-Avenue | ShanghaiTech | UCF-Crime | UBnormal |
|:------:|:----------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------:|:-------:|:---------:|:-----------:|:------------:|:---------:|:--------:|
|  BODS  |                                                       [Link](https://arxiv.org/abs/1908.05884)                                                       |                                   -                                    |   I3D   |    AE     |      -      |      -       |   68.2    |    -     |
|  GODS  |                                                       [Link](https://arxiv.org/abs/1908.05884)                                                       |                                   -                                    |   I3D   |    AE     |      -      |      -       |   69.4    |    -     |
|  VEC   |                                                       [Link](https://arxiv.org/abs/2008.11988)                                                       |         [Github Link](https://github.com/yuguangnudt/VEC_VAD)          |   OD    |  AE(FPM)  |    90.2     |     74.8     |     -     |    -     |
|  CAC   |                                              [Link](https://dl.acm.org/doi/abs/10.1145/3394171.3413529)                                              |                                   -                                    |   A3D   |  Encoder  |    87.0     |     79.3     |     -     |    -     |
|  HF2   |                                                       [Link](http://arxiv.org/abs/2108.06852)                                                        |           [Github Link](https://github.com/LiUzHiAn/hf2vad)            |   OD    |  AE(FPM)  |    91.1     |     76.2     |     -     |    -     |
|  BAF   |                                            [Link](https://ieeexplore.ieee.org/abstract/document/9410375)                                             |          [Github Link](https://github.com/lilygeorgescu/AED)           |   OD    |    AE     |    92.3     |     82.7     |     -     |   59.3   |
|  BDPN  |                                            [Link](https://ojs.aaai.org/index.php/AAAI/article/view/19898)                                            |                                   -                                    |   OD    |  AE(FPM)  |    90.0     |     78.1     |     -     |    -     |
|  GCL   |                                                       [Link](http://arxiv.org/abs/2203.03962)                                                        |                                   -                                    |   R3D   |    GAN    |      -      |     79.6     |   74.2    |    -     |
|  SSL   |                                                       [Link](https://arxiv.org/pdf/2207.10172)                                                       |         [Github Link](https://github.com/gdwang08/Jigsaw-VAD)          |   OD    |  Encoder  |    92.2     |     84.3     |     -     |    -     |
|        |                                            [Link](https://ieeexplore.ieee.org/abstract/document/10222594)                                            | [Github Link](https://github.com/AnilOsmanTur/video_anomaly_diffusion) |         |           |      -      |     76.1     |   65.2    |    -     |
|  FPDM  | [Link](https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.pdf) |          [Github Link](https://github.com/daidaidouer/FPDM)            |  Image  |   DDIM    |    90.1     |     78.6     |   74.7    |   62.7   |

#### Framework 
|            Type            |                       Framework                       |                                                                                       Paper                                                                                       |                               Github                                |
|:--------------------------:|:-----------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------:|
|   Unsupervised Learning    |          Future Frame Prediction Method(FPM)          |                                  [Link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Future_Frame_Prediction_CVPR_2018_paper.pdf)                                   |  [Github Link](https://github.com/StevenLiuWen/ano_pred_cvpr2018)   |
|   Unsupervised Learning    |                         MemAE                         | [Link](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf) | [Github Link](https://github.com/donggong1/memae-anomaly-detection) |
|   Unsupervised Learning    |                        AnoGAN                         |                                                 [Link](https://www.sciencedirect.com/science/article/pii/S1361841518302640)                                                       | [Github Link](https://github.com/A03ki/f-AnoGAN?tab=readme-ov-file) |
| Weakly Supervised Learning |            Multiple Instance Learning(MIL)            |                                                               [Link](https://ar5iv.labs.arxiv.org/html/2303.12369)                                                                |          [Github Link](https://github.com/ktr-hubrt/umil)           |
| Weakly Supervised Learning |     Regularized Two-Stream Feature Matching(RTFM)     |                                                               [Link](https://ar5iv.labs.arxiv.org/html/2101.10030)                                                                |          [Github Link](https://github.com/tianyu0207/RTFM)          |
| Weakly Supervised Learning | Clustering Assisted Weakly Supervised Learning(CLAWS) |                                                                     [Link](https://arxiv.org/abs/2011.12077)                                                                      |         [Github Link](https://github.com/xaggi/claws_eccv)          |


### OpticalFlow
- [Awesome-Optical-Flow](https://github.com/hzwer/Awesome-Optical-Flow?tab=readme-ov-file)

</details>
