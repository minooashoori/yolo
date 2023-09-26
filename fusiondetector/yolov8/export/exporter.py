# from ultralytics import YOLO
import sys
import os
sys.path.append("/home/ec2-user/dev/ctx-logoface-detector/ultralytics")
from ultralytics import YOLO
import torch
import yaml

class Exporter:

    def __init__(self) -> None:

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None


    def load(self, path:str=None):
        if path is None:
            path = "/home/ec2-user/dev/yolo/runs/detect/train51/weights/epoch59.pt" # default path

        model = YOLO(path, task="detect")
        # if  we load a model with .pt extension, we need to fuse it
        if path.endswith(".pt"):
            model.fuse()
            model.to(self.device)

        self.model = model
        print(f"Model device: {self.model.device}")
        return model

    def predict(self, img, imgsz=416):
        # source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
        results = self.model.predict(img, conf=0.01, device=self.device, imgsz=imgsz)
        return results

    def export(
        self,
        format:str,
        simplify:bool=True,
        device:str="0",
        half:bool=True,
        dynamic:bool=False,
        batch:int=32,
        metadata:dict=None):

        self.model.export(format=format,
                        simplify=simplify,
                        device=device,
                        half=half,
                        dynamic=dynamic,
                        batch=batch,
                        metadata=metadata,
                        )



if __name__ == "__main__":
    import os
    # read yaml metadata as dict
    with open("/home/ec2-user/dev/ctx-logoface-detector/fusiondetector/yolov8/export/metadata/yolov8s.yaml", 'r') as stream:
        metadata = yaml.safe_load(stream)
    print(metadata)
    exporter = Exporter()
    exporter.load("/home/ec2-user/dev/yolo/runs/detect/train7/weights/epoch5.pt")
    exporter.export(format="torchscript", half=True, dynamic=False, batch=32, device="cuda", simplify=True, metadata=metadata)

