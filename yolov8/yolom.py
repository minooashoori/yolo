from ultralytics import YOLO, settings
import wandb

# model = YOLO('yolov8m.yaml')
model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train39/weights/best.pt')


# # train the model with an s3 dataset
results = model.train(data="/home/ec2-user/dev/ctx-logoface-detector/yolov8/logo05fusion.yaml",
                      epochs=150,
                      batch=256,
                      imgsz=416,
                      mosaic=False,
                      device=[0,1,2, 3],
                      workers=32,
                      optimizer="SGD",
                      save_period=1,
                      warmup_epochs=0,
                      pretrained=True,
                      # fraction=0.1,
                    #   lr0=0.001,
                      save_conf=True,
                      save_crop=True,
                      augment=True,
                      half=True,
                      box=9.5,
                      # freeze=[i for i in range(0, 10)],
                      visualize=True
                      )
