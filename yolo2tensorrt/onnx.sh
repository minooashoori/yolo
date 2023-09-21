python3 export-det.py \                                                                                                                                                                    8s yolo ec2-user@ip-172-31-41-73
--weights /home/ec2-user/dev/ctx-logoface-detector/artifacts/yolov8s_t69_last.pt \
--iou-thres 0.45 \
--conf-thres 0.05 \
--topk 100 \
--opset 11 \
--sim \
--input-shape 32 3 416 416 \
--device cuda:0