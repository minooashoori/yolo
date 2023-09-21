import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from typing import List, Tuple, Union
from nms import non_max_suppression
# from compute_common.logger import logger
torch.set_printoptions(sci_mode=False)

class Detector:

    def __init__(
        self,
        model_path: str,
        confidence_thresholds: dict = None,
        detection_min_size_percentage: int = 0.04,
        batch_size: int = -1,
        fp16: bool = True,
        device: str = "cuda:0",
    ):

        self.class_map = {134533: 0, 90020: 1}

        # make sure the suffix is .torchscript
        jit = Path(model_path).suffix[1:] == "torchscript"
        if not jit:
            raise NotImplementedError(f'Model path suffix must be .torchscript got {model_path}')

        # check confidence thresholds
        self.conf_thres_classes = self._check_thresholds(confidence_thresholds, self.class_map)

        self.model, self.metadata = None, None
        self.fp16 = fp16

        self.device = torch.device(device)
        self.cuda = torch.cuda.is_available() and self.device.type != 'cpu'
        if not self.cuda:
            self.cuda = False

        self.model, self.metadata = self._load_jit(model_path, self.device, self.fp16)
        self.batch_size = batch_size


        self.imgsz = 416
        self.names = self.metadata['names']

        self.detection_min_size_percentage = detection_min_size_percentage

        self.warmup() # warmup the model

    @staticmethod
    def _load_jit(w: str, device: torch.device, fp16: bool) -> Tuple[torch.jit.ScriptModule, dict]:
        """
        Load a TorchScript model for inference.
        Args:
            w (str): path to TorchScript model
            device (torch.device): torch device to load the model on
            fp16 (bool): whether to use fp16 precision or not

        Returns:
            _type_: _description_
        """

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
    def _check_thresholds(conf_thrs: dict, class_map: dict) -> List[float]:

        conf_thrs = conf_thrs or {134533: 0.2, 90020: 0.1}

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

        im = [im for _ in range(self.batch_size)] if self.batch_size > 1 else [im]

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

        ims, n_ims = self._preprocess(ims)
        preds = self.predict(ims)
        detections = self._postprocess(preds, original_shapes, n_ims)
        
        #sanity check
        assert len(detections) == n_ims, f"Number of detections {len(detections)} does not match number of images {n_ims}"
        
        return detections


    def _preprocess(self, ims: List[np.ndarray]) -> torch.Tensor:
        """
        Prepares input image before inference.

        Args:
            ims (List[np.ndarray]): list of images in numpy format w/ format [h, w, c]
        Returns:
            torch.Tensor: BCHW tensor with images in the range [0, 1]
        """

        assert self._images_same_shapes(ims)

        n_ims = len(ims)
        batch_size = self.batch_size if self.batch_size != -1 else n_ims

        assert batch_size >= n_ims, f"Batch size must be greater than or equal to number of images or set as -1 for dynamic batch size."

        if batch_size > n_ims:
            n_ims_add = batch_size - n_ims
            black_im = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.float32) # HWC
            ims += [black_im for _ in range(n_ims_add)] # [B x [HWC]]

        # convert to torch tensor
        ims = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).contiguous() # BCHW
        ims = ims.to(self.device)
        ims = ims.half() if self.fp16 else ims.float() # uint8 to fp16/32
        # check if  we indeed are  using the right dtype
        assert ims.dtype == torch.float16 if self.fp16 else torch.float32
        ims /= 255.0 # 0 - 255 to 0.0 - 1.0
        return ims, n_ims


    def _filter(self,
                select: torch.Tensor,
                frames: torch.Tensor,
                boxes:torch.Tensor,
                scores: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        return frames[select], boxes[select], scores[select], labels[select]


    def _postprocess(self, preds, orig_shapes, n_ims):

        frames, boxes, scores, labels = non_max_suppression(preds, conf_thres_classes=self.conf_thres_classes)

        # remove detections that are not for the original images
        frames, boxes, scores, labels = self._filter(frames < n_ims, frames, boxes, scores, labels)

        # only run if we got boxes
        detections = None
        if boxes.numel() > 0:
            boxes = self._normalize_boxes(boxes)

            if orig_shapes:
                boxes = self._reshape_boxes(boxes, frames, orig_shapes)

            frames, boxes, scores, labels = self._ensure_min_size(frames, boxes, scores, labels)

            detections = self._stitch_results(frames, boxes, scores, labels, n_ims)


        return detections if detections else [[] for _ in range(n_ims)]

    def _stitch_results(self, frames: torch.Tensor, 
                        boxes: torch.Tensor, 
                        scores: torch.Tensor, 
                        labels: torch.Tensor, 
                        n_ims: int) -> List[List[Union[List[float], str, float]]]:
        """
        Stitch results together
        """

        # cast all tensors to numpy arrays on the cpu
        frames = frames.cpu().numpy()
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

        results = []
        for i in range(n_ims):
            select = np.equal(frames, i)
            res  = []
            for box, label, score in zip(boxes[select].tolist(), labels[select], scores[select]):
                res.append([box, str(label).encode("utf-8"), float(score)])
            results.append(res)
        return results

    def _ensure_min_size(self, frames, boxes, scores, labels) -> torch.Tensor:

        """
        Ensures that the already normalized boxes are at least % of the image size,
        """
        # if boxes are not of minimum size, remove them from boxes and remove the corresponding scores and labels and frames


        box_size_select = torch.logical_and((boxes[:, 2] - boxes[:, 0]) > self.detection_min_size_percentage,
                                            (boxes[:, 3] - boxes[:, 1]) > self.detection_min_size_percentage,
                                            )

        # filter the tensors
        boxes = boxes[box_size_select]
        scores = scores[box_size_select]
        labels = labels[box_size_select]
        frames = frames[box_size_select]

        return frames, boxes, scores, labels

    def _normalize_boxes(self, boxes: torch.Tensor) -> torch.Tensor:

            """
            Normalizes boxes between [0,1].
            """

            boxes /= self.imgsz
            # clamp boxes between 0 and 1
            boxes = boxes.clamp(0.0, 1.0)
            return boxes


    def _reshape_boxes(self, boxes: torch.Tensor, frames: torch.Tensor, orig_shapes: List[List[int]]) -> torch.Tensor:

        """
        Transforms the detections according to the frame original shape,
        and returns detections between [0,1].
        """

        if orig_shapes is None:
            return detections

        orig_shapes = torch.tensor(orig_shapes).to(frames.device)
        orig_shapes = orig_shapes[frames]


        resize_ratio = torch.min(self.imgsz / orig_shapes, dim=-1).values
        delta = (self.imgsz - (orig_shapes * resize_ratio.unsqueeze(1))) / 2
        abs_boxes = boxes * self.imgsz
        boxes[:, 0] = ((abs_boxes[:, 0] - delta[:, 1]) / resize_ratio) / orig_shapes[:, 1]
        boxes[:, 1] = ((abs_boxes[:, 1] - delta[:, 0]) / resize_ratio) / orig_shapes[:, 0]
        boxes[:, 2] = ((abs_boxes[:, 2] - delta[:, 1]) / resize_ratio) / orig_shapes[:, 1]
        boxes[:, 3] = ((abs_boxes[:, 3] - delta[:, 0]) / resize_ratio) / orig_shapes[:, 0]

        # clamp boxes between 0 and 1
        boxes = boxes.clamp(0.0, 1.0)
        return boxes



    def predict(self, im: torch.Tensor):

        y = self.model(im)
        return y


    def _images_same_shapes(self, ims: List[np.ndarray]) -> bool:

        return len({im.shape[:2] for im in ims}) == 1

    def _resize_boxes_original_shapes(self, detections: List[torch.Tensor], original_shapes: List[List[int]]) -> List[torch.Tensor]:

        pass


if __name__ == '__main__':

    detector = Detector(model_path="/home/ec2-user/dev/ctx-logoface-detector/artifacts/yolov8s_t74_best.torchscript", batch_size=-1)

    # read an image from path and convert it to numpy
    img_path = "/home/ec2-user/dev/data/logo05fusion/yolo/images/val/000000053.jpg"
    from PIL import Image
    img = Image.open(img_path)
    img = np.array(img)
    # print(img)
    img = [img for _ in range(3)]
    orig_shapes = [[400, 400], [800, 800], [600, 600]]
    detections = detector.detect(img, orig_shapes)
    print(detections)
