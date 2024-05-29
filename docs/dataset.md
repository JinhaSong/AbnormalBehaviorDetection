# Dataset

|   Dataset    | # of Class | # of testing video | # of training video | # of testing video frames | # of training video frames |
|:------------:|:----------:|:------------------:|:-------------------:|:-------------------------:|:--------------------------:|
| CUHK Avenue  |            |         21         |         16          |          15,324           |           15328            |
| ShanghaiTech |            |        107         |         330         |          40,791           |          274,515           |
|  UCF-Crime   |     16     |         -          |        1950         |             -             |         13,768,423         |
|   UBnormal   |            |                    |                     |                           |                            |

* __UCF-Crime classes__: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Normal_Videos_event, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Testing_Normal_Videos_Anomaly, Training_Normal_Videos_Anomaly, Vandalism

## CUHK-Avenue

<details><summary> <b>Expand</b> </summary>

```shell
CUHK_Avenue/
│   ├── testing/
│   │   ├── testing_videos/
│   │   │   ├── 01.avi
│   │   │   ├── ...
│   │   │   └── 21.avi
│   │   ├── testing_videos_frame
│   │   │   ├── 01/
│   │   │   ├── ...
│   │   │   └── 21/
│   │   └── testing_vol
│   └── training
│       ├── training_videos
│       │   ├── 01.avi
│       │   ├── ...
│       │   └── 16.avi
│       ├── training_videos_frame
│       │   ├── 01
│       │   ├── ...
│       │   └── 16
│       └── training_vol
└── ground_truth_demo
    ├── testing_label_mask
    └── testing_videos
```

</details>

## ShanghaiTech

<details><summary> <b>Expand</b> </summary>

```shell
/dataset/ShanghaiTech/
├── testing
│   ├── frames
│   │   ├── 01_0014
│   │   ├── ...
│   │   ├── 01_0177
│   │   ├── 02_0128
│   │   ├── ...
│   │   ├── 02_0164
│   │   ├── 03_0031
│   │   ├── ...
│   │   ├── 03_0061
│   │   ├── 04_0001
│   │   ├── ...
│   │   ├── 04_0050
│   │   ├── 05_0017
│   │   ├── ...
│   │   ├── 05_0024
│   │   ├── 06_0144
│   │   ├── ...
│   │   ├── 06_0155
│   │   ├── 07_0005
│   │   ├── ...
│   │   ├── 07_0049
│   │   ├── 08_0044
│   │   ├── ...
│   │   ├── 08_0179
│   │   ├── 09_0057
│   │   ├── 10_0037
│   │   ├── ...
│   │   ├── 10_0075
│   │   ├── 11_0176
│   │   ├── 12_0142
│   │   ├── ...
│   │   └── 12_0175
│   ├── test_frame_mask
│   └── test_pixel_mask
└── training
    ├── frames
    │   ├── 01_001
    │   ├── ...
    │   ├── 01_083
    │   ├── 02_001
    │   ├── ...
    │   ├── 02_015
    │   ├── 03_001
    │   ├── ...
    │   ├── 03_006
    │   ├── 04_001
    │   ├── ...
    │   ├── 04_020
    │   ├── 05_001
    │   ├── ...
    │   ├── 06_001
    │   ├── ...
    │   ├── 06_030
    │   ├── 07_001
    │   ├── ...
    │   ├── 08_001
    │   ├── ...
    │   ├── 08_049
    │   ├── 09_001
    │   ├── ...
    │   ├── 09_010
    │   ├── 10_001
    │   ├── ...
    │   ├── 10_011
    │   ├── 11_001
    │   ├── ...
    │   ├── 11_010
    │   ├── 12_001
    │   ├── ...
    │   ├── 12_015
    │   ├── 13_001
    │   ├── ...
    │   └── 13_007
    └── videos
```

</details>

## UCF-Crime

<details><summary> <b>Expand</b> </summary>

```shell
/dataset/UCF-Crime/
├── Action_Regnition_splits
├── Anomaly_Detection_splits
├── Videos
│   ├── Abuse
│   ├── Arrest
│   ├── Arson
│   ├── Assault
│   ├── Burglary
│   ├── Explosion
│   ├── Fighting
│   ├── Normal_Videos_event
│   ├── RoadAccidents
│   ├── Robbery
│   ├── Shooting
│   ├── Shoplifting
│   ├── Stealing
│   ├── Testing_Normal_Videos_Anomaly
│   ├── Training_Normal_Videos_Anomaly
│   └── Vandalism
└── frames
    ├── Abuse001_x264/
    ├── ...
    └── Vandalism050_x264/
    
```

</details>

## UBnormal
