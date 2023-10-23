import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"ultralytics"))
from ultralytics import YOLO, settings
import wandb


model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train51/weights/epoch59.pt')

results = model.train(data="/home/ec2-user/dev/ctx-logoface-detector/yolov8/logo05fusion.yaml",
                      epochs=40,
                      batch=256,
                      imgsz=416,
                      mosaic=False,
                      close_mosaic=0,
                      device=[0,1,2,3],
                      workers=32,
                      optimizer="Adam",
                      save_period=1,
                    #   warmup_epochs=3,
                      pretrained=True,
                      lr0=0.00001,
                      save_conf=True,
                      save_crop=True,
                      augment=True,
                      half=True,
                      box=9.5,
                      verbose=True,
                      save_json=False,
                      # weight_decay=	0.0005*2,
                      freeze=[i for i in range(0, 2)],
                      visualize=True,
                      # translate=0.1,
                      # scale=0.1,
                      degrees=5.0,
                      flipud=0.005,
                      )
# dbfs:/Shared/MLS/req_updated_ds_dev_ph.sh