import os
from utils.boxes import plot_boxes

plot_boxes("/home/ec2-user/dev/data/logo05/yolo/images/test/43557144.jpg",
           annotation="/home/ec2-user/dev/data/logo05/annotations/gts_preds/preds_yolo/43557144.txt",
           box_type="xywh",
           save=True)