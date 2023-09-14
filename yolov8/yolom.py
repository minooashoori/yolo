from ultralytics import YOLO, settings
import wandb

# model = YOLO('yolov8m.yaml')
# model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train39/weights/best.pt')
# model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train61/weights/last.pt')
model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train51/weights/epoch59.pt')


# results = model.train(resume=True)

results = model.train(data="/home/ec2-user/dev/ctx-logoface-detector/yolov8/portal.yaml",
                      epochs=10,
                      batch=256,
                      imgsz=416,
                      mosaic=False,
                      device=[0,1,2,3],
                      workers=32,
                      optimizer="SGD",
                      save_period=1,
                      warmup_epochs=0,
                      pretrained=True,
                      # fraction=0.1,
                      lr0=0.005,
                      save_conf=True,
                      save_crop=True,
                      augment=True,
                      half=True,
                    #   box=9.5,
                      box=12.5,
                      freeze=[i for i in range(0, 10)],
                      visualize=True
                      )
