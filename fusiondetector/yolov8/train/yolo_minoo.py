import sys
import os
from ultralytics import YOLO, settings
# import wandb

model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolos')

results = model.train(data="/home/ubuntu/ctx-logoface-detector/fusiondetector/yolov8/datasets/minoo.yaml",
                      epochs=2,
                      batch =256,
                      imgsz=416,
                      mosaic=False,
                      close_mosaic=10, 
                      device=[0,1,2,3],
                      workers=32,
                      optimizer="SGD",
                      # optimizer="Adam",
                      save_period=1,
                      # warmup_epochs=3,
                      pretrained=True,
                      # lr0=0.00005,# adam
                      lr0=0.00025, # sgd
                      save_conf=True,
                      save_crop=True,
                      augment=True,
                      half=True,
                    #   box=9.5,
                      verbose=True,
                      save_json=False,
                      # weight_decay=	0.0005*2,
                      # freeze=[i for i in range(0, 5)],
                      visualize=True,
                      # translate=0.1,
                      # scale=0.1,
                      degrees=5.0, #was 5
                      flipud=0.005,
                      )

if results is None:
  pass
else:
  print('khar')
  print(results)
  print('khar')
#   print(results.results_dict['metrics/mAP50(B)'])
  # results_dict = results.get('results_dict', None)
  # print(results_dict)