
# Performance Logo/Face Detector YOLOv8

## Commands:

# Logo

```bash
python pascalvoc.py --gt /home/ec2-user/dev/data/logo05/annotations/gts_preds/gts --det /home/ec2-user/dev/data/logo05/annotations/gts_preds/preds_yolo_small  -t 0.45 -conf 0.2
```
# Face

```bash
python pascalvoc.py --gt /home/ec2-user/dev/data/widerface/gts_preds/gts --det /home/ec2-user/dev/data/widerface/gts_preds/preds_yolo_small -t 0.45 -conf 0.2
```


# Production Models

## IOU

- IOU Threshold: 0.45

## Sep/2023 (YOLOv4)

### Logo (Threshold: 0.35)

- Class: '1'
  - Precision: 0.59
  - Recall: 0.72
- Average Precision (AP): 63.94%

### Face (Threshold: 0.4)

- Class: '0'
  - Precision: 0.83
  - Recall: 0.46
- Average Precision (AP): 44.66%

# Trained Models
## Train69 (YOLOv8s) ** problem with false detections of logos on plain images!

### Logo (Threshold: 0.25)
- Class: '1'
  - Precision: 0.61
  - Recall: 0.73
- Average Precision (AP): 71.47% 

### Face (Threshold: 0.05)
- Class: '0'
  - Precision: 0.8
  - Recall: 0.44
- Average Precision (AP): 46.34% (0)

## Train74 (Yolov8s) ** problem with false detections of logos on plain images!

### Logo (Threshold: 0.1)

- Class: '1'
  - Precision: 0.6
  - Recall: 0.72
- Average Precision (AP): 68.94%

### Face (Threshold: 0.2)

- Class: '0'
  - Precision: 0.82
  - Recall: 0.44
- Average Precision (AP): 48.61%

## Train82 (Yolov8s)

### Logo (mAP: 71.01%)

- Threshold: 0.2

    - Precision: 0.67
    - Recall: 0.7

- Threshold: 0.15

    - Precision: 0.62
    - Recall: 0.73

### Face (mAP: 49.00%)

- Threshold: 0.2
    - Precision: 0.89
    - Recall: 0.41

- Threshold: 0.15
    - Precision: 0.85
    - Recall: 0.43

## Train103 (Yolov8s) Epoch 29

### Logo

- Threshold: 0.15
  - mAP: 70.17%
  - Class: '1'
    - Precision: 0.63
    - Recall: 0.72

- Confidence (conf): 0.2
  - mAP: 70.17%
  - Class: '1'
    - Precision: 0.68
    - Recall: 0.7

### Face

- Confidence (conf): 0.2
  - mAP: 47.90%
  - Class: '0'
    - Precision: 0.9
    - Recall: 0.4

- Confidence (conf): 0.15
  - mAP: 47.90%
  - Class: '0'
    - Precision: 0.86
    - Recall: 0.42

- Confidence (conf): 0.1
  - mAP: 47.90%
  - Class: '0'
    - Precision: 0.8
    - Recall: 0.44

## Train104 (Yolov8s) Last.pt

### Logo (AP: 70.67%)

- Confidence (conf): 0.2
    - Precision: 0.7
    - Recall: 0.7

- Confidence (conf): 0.15

    - Precision: 0.65
    - Recall: 0.72


- Confidence (conf): 0.1
    - Precision: 0.59
    - Recall: 0.74

### Face (AP: 48.40%)

- Confidence (conf): 0.2
    - Precision: 0.89
    - Recall: 0.41

- Confidence (conf): 0.15
    - Precision: 0.85
    - Recall: 0.43

- Confidence (conf): 0.1
    - Precision: 0.79
    - Recall: 0.45

## Train105 (Yolov8s) Best.pt

### Logo (AP: 70.83%)

- Confidence (conf): 0.2
    - Precision: 0.69
    - Recall: 0.7

- Confidence (conf): 0.15

    - Precision: 0.64
    - Recall: 0.72

- Confidence (conf): 0.1

    - Precision: 0.58
    - Recall: 0.74

### Face (AP: 49.25%)

- Confidence (conf): 0.2
  - Precision: 0.87
  - Recall: 0.42

- Confidence (conf): 0.15
  - Precision: 0.82
  - Recall: 0.44

- Confidence (conf): 0.1
  - Precision: 0.75
  - Recall: 0.46

## Train0 (Yolov8M) - epoch4

### Logo (AP: 72.04%)

- Confidence (conf): 0.2
    - Precision: 0.69
    - Recall: 0.7

- Confidence (conf): 0.15

    - Precision: 0.62
    - Recall: 0.73



### Face (AP: 49.42%)

- Confidence (conf): 0.2
  - Precision: 0.89
  - Recall: 0.42

- Confidence (conf): 0.15
  - Precision: 0.85
  - Recall: 0.44

- Confidence (conf): 0.1
  - Precision: 0.79
  - Recall: 0.46