
from ultralytics import YOLO, settings
import os

model = YOLO('yolov8n.yaml').load('/home/ec2-user/dev/yolo/runs/detect/train17/weights/last.pt') # or by string name

# # train the model with an s3 dataset
results = model.train(data="/home/ec2-user/dev/ctx-logoface-detector/yolov8/yolo_logodet3k.yaml", epochs=10, batch=128, imgsz = 320, mosaic=False, visualize=True)

# a change



