from ultralytics import YOLO, settings
import wandb

model = YOLO('yolov8m.yaml')
model = YOLO('yolov8m.pt')


# # train the model with an s3 dataset
results = model.train(data="/home/ec2-user/dev/ctx-logoface-detector/yolov8/totalfusion_logodet3k.yaml",
                      epochs=150,
                      batch=128,
                      imgsz=416,
                      mosaic=False,
                      device=[0,1,2, 3],
                      workers=32,
                      visualize=True)
