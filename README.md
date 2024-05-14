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
- [yolov4](https://github.com/AlexeyAB/darknet)
- [yolov7](https://github.com/WongKinYiu/yolov7)
- [yolov8](https://github.com/ultralytics/ultralytics)
- [yolov9](https://github.com/WongKinYiu/yolov9)

### Object Tracking
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [SORT](https://github.com/abewley/sort)

### Pose Estimation
- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
- [MMPose](https://github.com/open-mmlab/mmpose)

### Action Recognition
- [MMAction2](https://github.com/open-mmlab/mmaction2)

### Abnormal Event Detection(Real-world Anomaly Detection in Surveillance Videos)
#### Violence(Assault, Fight)
- [PapersWidthCode](https://paperswithcode.com/paper/real-world-anomaly-detection-in-surveillance)
- [UBI-Fights](https://paperswithcode.com/sota/abnormal-event-detection-in-video-on-ubi)
- [UCF-Crime](https://paperswithcode.com/sota/anomaly-detection-in-surveillance-videos-on)

#### Fall

- 

#### Wander

- 


### OpticalFlow
- [Awesome-Optical-Flow](https://github.com/hzwer/Awesome-Optical-Flow?tab=readme-ov-file)

</details>
