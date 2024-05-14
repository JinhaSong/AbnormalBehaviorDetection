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

## Evaluation

## Inference


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
 ├─ tools/
 ├─ scripts/
 ├─ utils/
 └─ docker-compose.yml
```

- docker: docker 빌드를 위한 dockerfile, env 및 requirements.txt 파일 포함
- lib: git submodule로 추가한 관련 라이브러리 및 repository
- models: abnormal behavior detection 모델 class
- tools: 테스트용 소스 및 툴
- scripts: 데이터셋 및 모델 다운로드용 shell scripts

</details>

## Note
<details><summary> <b>Expand</b> </summary>

### OpticalFlow
- [Awesome-Optical-Flow](https://github.com/hzwer/Awesome-Optical-Flow?tab=readme-ov-file)

</details>
