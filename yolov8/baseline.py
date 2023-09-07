from ultralytics import YOLO, settings
import wandb




# settings.update({"datasets_dir": "/home/ec2-user/dev/"})
# print(settings)
# s3 = s3fs.S3FileSystem()

# # load a model
# # model = YOLO('yolov8n.yaml').load('/home/ec2-user/dev/yolo/runs/detect/train11/weights/last.pt') # or by string name


model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train36/weights/best.pt')


# # train the model with an s3 dataset
results = model.train(data="/home/ec2-user/dev/ctx-logoface-detector/yolov8/totalfusion_logodet3k.yaml",
                      epochs=50,
                      batch=128,
                      imgsz=416,
                      mosaic=False,
                      device=[0,1,2, 3],
                      workers=32,
                      visualize=True)

# results = model.train(data=data, epochs=100, imgsz=640)ls
