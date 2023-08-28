from ultralytics import YOLO, settings
import wandb

import s3fs



# settings.update({"datasets_dir": "/home/ec2-user/dev/"})
# print(settings)
# s3 = s3fs.S3FileSystem()

# # load a model
# # model = YOLO('yolov8n.yaml').load('/home/ec2-user/dev/yolo/runs/detect/train11/weights/last.pt') # or by string name


model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train29/weights/best.pt')


# # train the model with an s3 dataset
results = model.train(data="/home/ec2-user/dev/ctx-logoface-detector/yolov8/yolo200k.yaml",
                      epochs=50,
                      batch=128,
                      imgsz=320,
                      mosaic=False, 
                      visualize=True)

# results = model.train(data=data, epochs=100, imgsz=640)ls

#