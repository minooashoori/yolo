import sys
import os
# from fusiondetector import PROJECT_DIR
sys.path.append("/home/ec2-user/dev/ctx-logoface-detector/ultralytics")
from ultralytics import YOLO, settings
import wandb

# model = YOLO('yolov8m.yaml')
# model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train39/weights/best.pt')
# model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train61/weights/last.pt')
# model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train51/weights/epoch59.pt')
# model = YOLO('yolov8s.yaml')
# model =  YOLO('/home/ec2-user/dev/ctx-logoface-detector/artifacts/yolov8s_t77_last.pt')
# model = YOLO('/home/ec2-user/dev/ctx-logoface-detector/artifacts/yolov8s_t82_best.pt')
model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train7/weights/epoch5.pt')


# results = model.train(resume=True)

results = model.train(data="/home/ec2-user/dev/ctx-logoface-detector/fusiondetector/yolov8/datasets/newfusionlogodet3k.yaml",
                      epochs=100,
                      batch=256,
                      imgsz=416,
                      mosaic=True,
                      close_mosaic=8,
                      device=[0,1,2,3],
                      workers=32,
                      optimizer="SGD",
                      save_period=1,
                    #   warmup_epochs=3,
                      pretrained=True,
                      # lr0=0.00005,# adam
                      # lr0=0.0005, # sgd
                      save_conf=True,
                      save_crop=True,
                      augment=True,
                      half=True,
                      box=9.5,
                      verbose=True,
                      save_json=False,
                      # weight_decay=	0.0005*2,
                      # freeze=[i for i in range(0, 2)],
                      visualize=True,
                      # translate=0.1,
                      # scale=0.1,
                      degrees=5.0, #was 5
                      flipud=0.005,
                      )
