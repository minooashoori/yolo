import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from typing import List, Tuple
from nms import non_max_suppression
# from compute_common.logger import logger
torch.set_printoptions(sci_mode=False)

class Detector:

    def __init__(
        self,
        model_path: str,
        confidence_thresholds: dict = None,
        detection_min_size_percentage: int = 0.1,
        batch_size: int = None,
        process_gpu_memory_fraction: float = 1.0,
        fp16: bool = True,
        device: str = "cuda:0",
    ):

        class_map = {134533: 0, 90020: 1}

        # make sure the suffix is .torchscript
        jit = Path(model_path).suffix[1:] == "torchscript"
        if not jit:
            raise NotImplementedError(f'Model path suffix must be .torchscript got {model_path}')

        # check confidence thresholds
        conf_thres_classes = self._check_thresholds(confidence_thresholds, class_map)

        model, metadata = None, None

        device = torch.device(device)
        cuda = torch.cuda.is_available() and device.type != 'cpu'
        if not cuda:
            cuda = False

        model, metadata = self._load_jit(model_path, device, fp16)

        batch_size = batch_size or metadata['batch']
        imgsz = 416
        names = metadata['names']

        detection_min_size_percentage = detection_min_size_percentage or 0.1
        process_gpu_memory_fraction = process_gpu_memory_fraction or 1.0

        self.__dict__.update(locals())  # assign all variables to self

        self.warmup() # warmup the model

    @staticmethod
    def _load_jit(w, device, fp16):
        # logger.info(f'Loading {w} for TorchScript Runtime inference...')
        print(f'Loading {w} for TorchScript Runtime inference...')
        extra_files = {'config.txt': ''}  # model metadata
        model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
        model.half() if fp16 else model.float()
        model.eval()
        if extra_files['config.txt']:  # load metadata dict
            metadata = json.loads(extra_files['config.txt'], object_hook=lambda x: dict(x.items()))
        return model, metadata

    @staticmethod
    def _check_thresholds(conf_thrs: dict, class_map: dict) -> None:

        conf_thrs = conf_thrs or {134533: 0.3, 90020: 0.1}

        conf_thrs = {int(k): v for k, v in conf_thrs.items()}
        # confidence thresholds and categories map must have the same keys
        if set(conf_thrs.keys()) != set(class_map.keys()):
            raise ValueError(
                f"Confidence thresholds must have keys: 134533 (face) and 90020 (logo) but got {conf_thrs.keys()}"
            )
        # logger.info(f"Confidence thresholds: {conf_thrs}")
        print(f"Confidence thresholds: {conf_thrs}")
        # sort the conf thresholds by the values of class_map and return a list
        conf_thrs = [conf_thrs[k] for k in sorted(conf_thrs.keys(), key=lambda x: class_map[x])]
        print(f"Confidence thresholds: {conf_thrs}")

        return conf_thrs

    def warmup(self) -> None:
        # logger.info("Warming up the detector model...")
        print("Warming up the detector model...")
        im = np.array(np.random.uniform(0, 255, [self.imgsz, self.imgsz, 3]), dtype=np.float32)
        im = [im for _ in range(self.batch_size)]
        self.detect(im)
        # logger.info("Warmup finished.")
        print("Warmup finished.")

    def detect(self, ims: List[np.ndarray], original_shapes: List[List[int]] = None):
        """
        Detect objects in images.
        Args:
            ims: list of images in numpy format w/ original format [h, w, c]
            original_shape: list of original image shapes
        """

        ims = self._preprocess(ims)
        res = self.predict(ims)
        return res


    def _preprocess(self, ims: List[np.ndarray]) -> torch.Tensor:
        """
        Prepares input image before inference.

        Args:
            ims (List[np.ndarray]): list of images in numpy format w/ format [h, w, c]
        Returns:
            torch.Tensor: BCHW tensor
        """

        assert self._images_same_shapes(ims)

        n_ims = len(ims)

        if self.batch_size:
            n_ims_add = self.batch_size - n_ims
            black_im = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.float32) # HWC
            ims += [black_im for _ in range(n_ims_add)] # [B x [HWC]]

        # convert to torch tensor
        ims = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).contiguous() # BCHW
        ims = ims.to(self.device)
        ims = ims.half() if self.fp16 else ims.float() # uint8 to fp16/32
        ims /= 255 # 0 - 255 to 0.0 - 1.0
        return ims


    def _postprocess(self, preds, orig_shapes):

        detections = []
        detections = non_max_suppression(preds, 
                                        conf_thres = float(min(self.conf_thres_classes)),
                                        conf_thres_classes=self.conf_thres_classes)



    def predict(self, im: torch.Tensor):

        y = self.model(im)
        return y


    def _images_same_shapes(self, ims: List[np.ndarray]) -> bool:

        return len({im.shape[:2] for im in ims}) == 1

    def _resize_boxes_original_shapes(self, detections: List[torch.Tensor], original_shapes: List[List[int]]) -> List[torch.Tensor]:

        pass




if __name__ == '__main__':
    
    detector = Detector(model_path="/home/ec2-user/dev/yolo/runs/detect/train69/weights/last.torchscript", batch_size=1)

    # read an image from path and convert it to numpy
    img_path = "/home/ec2-user/dev/data/logo05fusion/yolo/images/val/000000053.jpg"
    from PIL import Image
    img = Image.open(img_path)
    img = np.array(img)
    # print(img)
    img = [img for _ in range(1)]
    detections = detector.detect(img)
    # print(detections[0, :, 0])
    print(detections.shape)
    non_max_suppression(detections, conf_thres_classes=(0.8, 0.2))
    
    


    # def postprocess(self, preds, img, orig_imgs):
    #     """Post-processes predictions and returns a list of Results objects."""
    #     preds = ops.non_max_suppression(preds,
    #                                     self.args.conf,
    #                                     self.args.iou,
    #                                     agnostic=self.args.agnostic_nms,
    #                                     max_det=self.args.max_det,
    #                                     classes=self.args.classes)

    #     if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
    #         orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    #     results = []
    #     for i, pred in enumerate(preds):
    #         orig_img = orig_imgs[i]
    #         pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
    #         img_path = self.batch[0][i]
    #         results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
    #     return results


    # def preprocess(self, im):
    #     """Prepares input image before inference.

    #     Args:
    #         im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
    #     """
    #     not_tensor = not isinstance(im, torch.Tensor)
    #     if not_tensor:
    #         im = np.stack(self.pre_transform(im))
    #         im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    #         im = np.ascontiguousarray(im)  # contiguous
    #         im = torch.from_numpy(im)

    #     im = im.to(self.device)
    #     im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
    #     if not_tensor:
    #         im /= 255  # 0 - 255 to 0.0 - 1.0
    #     return im
