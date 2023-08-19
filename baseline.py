from ultralytics import YOLO, settings
import s3fs



# settings.update({"datasets_dir": "/home/ec2-user/dev/"})
# print(settings)
# s3 = s3fs.S3FileSystem()

# # load a model
# # model = YOLO('yolov8n.yaml').load('/home/ec2-user/dev/yolo/runs/detect/train11/weights/last.pt') # or by string name

# model = YOLO('yolov8n.yaml').load('/home/ec2-user/dev/yolo/runs/detect/train17/weights/last.pt')

model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train17/weights/last.pt')

# # train the model with an s3 dataset
results = model.train(data="yolo60k.yaml", epochs=300, batch=128, imgsz = 320, mosaic=False, visualize=True)

# results = model.train(data=data, epochs=100, imgsz=640)ls