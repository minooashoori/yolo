from ultralytics import YOLO
import s3fs

s3 = s3fs.S3FileSystem()

# load a model
model = YOLO('yolov8n.yaml').load('yolov8n.pt') # or by string name

# train the model with an s3 dataset
results = model.train(data="yolo200k.yaml", epochs=200, amp=True)

# results = model.train(data=data, epochs=100, imgsz=640)ls