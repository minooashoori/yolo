import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"ultralytics"))
from ultralytics import YOLO, settings
import wandb

# model = YOLO('yolov8m.yaml')
# model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train39/weights/best.pt')
# model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train61/weights/last.pt')
# model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train51/weights/epoch59.pt')
model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train66/weights/best.pt')


# results = model.train(resume=True)

results = model.train(data="/home/ec2-user/dev/ctx-logoface-detector/yolov8/logo05fusion.yaml",
                      epochs=50,
                      batch=256,
                      imgsz=416,
                      mosaic=False,
                      device=[0,1,2,3],
                      workers=32,
                      optimizer="SGD",
                      save_period=1,
                      warmup_epochs=1,
                      pretrained=True,
                      # fraction=0.1,
                    #   lr0=0.005,
                      save_conf=True,
                      save_crop=True,
                      augment=True,
                      half=True,
                      box=9.5,
                      verbose=True,
                      save_json=True,
                    #   freeze=[i for i in range(0, 10)],
                      visualize=True
                      )
