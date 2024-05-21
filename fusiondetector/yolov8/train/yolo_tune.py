from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('yolov8n.yaml')

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(data='/home/ubuntu/ctx-logoface-detector/fusiondetector/yolov8/datasets/minoo.yaml', 
           epochs=50, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)


