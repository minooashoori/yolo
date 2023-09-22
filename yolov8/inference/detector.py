import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import pprint
from nms import non_max_suppression

# from compute_common.logger import logger
torch.set_printoptions(sci_mode=False)
pp = pprint.PrettyPrinter(indent=4)

class FusionFaceLogoDetector:

    """
    Detector class for object detection of faces and logos using a TorchScript model.

    Args:
        model_path (str): The path to the TorchScript model file.
        conf_thrs (Union[dict, float]): Confidence thresholds for face and logo detection.
        box_min_perc (float): The minimum percentage of image size for a detected bounding box.
        batch_size (int): The batch size for inference. Set to -1 for dynamic batch size.
        fp16 (bool): Whether to use 16-bit floating-point precision (half-precision) for inference.
        device (str): The target torch device where the model should be loaded ("cuda:0" for GPU or "cpu" for CPU).
        max_det (int): The maximum number of boxes to keep after NMS.

    Notes:
        - This class serves as a detector for faces and logos using a TorchScript model.
        - It provides methods for detecting objects in images, model warm-up, and other utility functions.
        - Confidence thresholds can be specified as a float or a dictionary with keys 134533/face and 90020/logo.
        - The batch size can be set to -1 for dynamic batch size based on the number of input images. Careful with this option as it can lead to out of memory errors.
    """

    def __init__(
        self,
        model_path: str,
        conf_thrs: Union[dict, float] = None,
        box_min_perc: float = 0.04,
        batch_size: int = -1,
        fp16: bool = True,
        device: str = "cuda:0",
        max_det: int = 100,
    ):
        # set the ids and names
        self.ids = {0: "134533", 1: "90020"}
        self.names = {0: "face", 1: "logo"}

        jit = Path(model_path).suffix[1:] == "torchscript"
        if not jit:
            raise NotImplementedError(f'Model path suffix must be .torchscript got {model_path}')

        # check confidence thresholds
        self.conf_thrs = self._check_thresholds(conf_thrs)
        # print(f"Confidence thresholds: {self.conf_thrs}")

        # set the device and fp16
        self.device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.fp16 = fp16 if self.device.type != 'cpu' else False

        # load the model and metadata
        self.model, self.metadata = self._load_jit(model_path, self.device, self.fp16)
        # pp.pprint(self.metadata)

        # check the batch size  - batch size cannot be 0 or lower than -1
        self.batch_size = batch_size if batch_size > 0 or batch_size == -1 else 1

        self.imgsz = 416
        self.box_min_perc = box_min_perc
        self.max_det = max_det

        self.warmup() # warmup the model

    @staticmethod
    def _load_jit(w: str, device: torch.device, fp16: bool) -> Tuple[torch.jit.ScriptModule, dict]:
        """
        Load a TorchScript model for inference.
        Args:
            w (str): The path to the TorchScript model file.
            device (torch.device): The target torch device where the model should be loaded (e.g., "cuda:0" for GPU or "cpu" for CPU).
            fp16 (bool):  Whether to use 16-bit floating-point precision (half-precision) for inference, which can be more memory-efficient.

        Returns:
            Tuple[torch.jit.ScriptModule, dict]: A tuple containing the loaded model as a TorchScript ScriptModule and a dictionary
            containing metadata associated with the model (e.g., model configuration).

        Example:
            model, metadata = Detector._load_jit("model.torchscript", torch.device("cuda:0"), fp16=True)
        """

        # logger.info(f'Loading {w} for TorchScript Runtime inference...')
        # print(f'Loading {w} for TorchScript Runtime inference...')
        extra_files = {'config.txt': ''}  # model metadata
        model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
        model.half() if fp16 else model.float()
        model.eval()
        if extra_files['config.txt']:  # load metadata dict
            metadata = json.loads(extra_files['config.txt'], object_hook=lambda x: dict(x.items()))
        return model, metadata

    @staticmethod
    def _check_thresholds(conf_thrs: Union[dict, float]) -> List[float]:
        """
        Check and transform confidence thresholds.

        Args:
            conf_thrs (Union[dict, float]): Confidence thresholds can have 3 formats:
                1. None
                2. float
                3. Dictionary with keys 134533 and 90020, "134533" and "90020" or face and logo
        Returns:
            List[float]: List of confidence threshold values in the order [face, logo].
        Raises:
            NotImplementedError: If the input format is not supported.
        """
        conf_thrs = conf_thrs if conf_thrs is not None else {134533: 0.2, 90020: 0.1}

        # if it's a float, then just output the list of floats
        if isinstance(conf_thrs, float):
            # check if it's a valid float
            if not 0.0 <= conf_thrs <= 1.0:
                raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {conf_thrs}")

            return [conf_thrs for _ in range(2)]

        elif isinstance(conf_thrs, dict):

            # if the keys are face and logo, convert them to 134533 and 90020
            if set(conf_thrs.keys()) == set(["face", "logo"]):

                conf_thrs = [conf_thrs["face"], conf_thrs["logo"]]
            elif set(conf_thrs.keys()) in [set(["134533", "90020"]),set([134533, 90020])]:
                # set key to int
                conf_thrs = {int(k): v for k, v in conf_thrs.items()}
                conf_thrs = [conf_thrs[134533], conf_thrs[90020]]
            else:
                raise NotImplementedError(f"Confidence thresholds keys must be: [face, logo], [134533, 90020], got {set(conf_thrs.keys())}")
        else:
            raise NotImplementedError(f"Confidence thresholds must be either a float, a dict or None, got {type(conf_thrs)}")
        # check if the values are valid
        if not all([0.0 <= t <= 1.0 for t in conf_thrs]):
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {conf_thrs}")

        # extract the list of thresholds
        return conf_thrs


    def warmup(self) -> None:
        """
        Warm up the detector model.

        This method prepares the detector model for inference by generating random input images and running
        them through the detector. It ensures that the model is loaded and initialized properly.

        Returns:
            None
        """

        # logger.info("Warming up the detector model...")
        # print("Warming up the detector model...")
        im = np.array(np.random.uniform(0, 255, [self.imgsz, self.imgsz, 3]), dtype=np.float32)

        im = [im for _ in range(self.batch_size)] if self.batch_size > 1 else [im]

        self.detect(im)
        # logger.info("Warmup finished.")
        # print("Warmup finished.")

    def detect(self, ims: List[np.ndarray], original_shapes: List[List[int]] = None) -> List[List[Union[List[float], bytes, float]]]:
        """
        Detect objects in a list of images.

        Args:
            ims (List[np.ndarray]): A list of images in numpy format with original dimensions [H, W, C].
            original_shapes (List[List[int]], optional): A list of original image shapes in the format [H, W].

        Returns:
            List[List[Union[List[float], bytes, float]]]: A list of detection results for each image.
                Each innermost list contains information for one detection, including the bounding box coordinates in the xyxy format,
                class label (as a string encoded in UTF-8), and confidence score.

        Notes:
            - The `original_shapes` parameter is optional and can be used to provide the original dimensions of the images.
            - Before detection, the images are preprocessed, and after detection, the results are post-processed.

        Raises:
            AssertionError: If the number of detections does not match the number of input images.
        """

        if not isinstance(ims, list):
            raise NotImplementedError(f"Input must be a list of images, got {type(ims)}")

        if len(ims) == 0:
            return []

        ims, n_ims = self._preprocess(ims)
        preds = self.model(ims)
        detections = self._postprocess(preds, n_ims, original_shapes)

        #sanity check
        assert len(detections) == n_ims, f"Number of detections {len(detections)} does not match number of images {n_ims}"

        return detections


    def _preprocess(self, ims: List[np.ndarray]) -> Tuple[torch.Tensor, int]:
        """
            Preprocesses a list of images for inference.

            Args:
                ims (List[np.ndarray]): A list of images in numpy format with dimensions [height, width, channels].

            Returns:
                torch.Tensor: A BCHW tensor with images in the range [0, 1].
                n_ims (int): The number of (real) images in the batch.

            Raises:
                AssertionError: If the input images have different shapes or if batch size is insufficient.

            Notes:
                - This method ensures that all input images have the same shape and prepares them for inference.
                - It also manages the batch size and data types to match the model's requirements.
        """

        # Ensure that all input images have the same shape
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

        """
        Filter and select elements from input tensors based on a selection mask.

        Args:
            select (torch.Tensor): A boolean tensor indicating which elements to select.
            frames (torch.Tensor): A tensor representing frames or images.
            boxes (torch.Tensor): A tensor containing bounding boxes in the format [x1, y1, x2, y2].
            scores (torch.Tensor): A tensor containing confidence scores.
            labels (torch.Tensor): A tensor containing class labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Selected elements from the input tensors.
        """
        return frames[select], boxes[select], scores[select], labels[select]


    def _postprocess(self, preds, n_ims, orig_shapes=None):

        """
        Post-processes detection results to organize and filter detections.

        Args:
            preds: The raw predictions from the model in the format (batch_size, num_classes + 4, num_boxes).
            n_ims (int): The number of (real) images in the batch.
            orig_shapes (List[List[int]], optional): A list of original image shapes with dimensions [H, W].

        Returns:
            List[List[Union[List[float], bytes, float]]]: A list of organized detection results for each image.
                Each inner list contains information for one detection, including the bounding box coordinates,
                class label (as a string encoded in UTF-8), and confidence score.

        Notes:
            - This method applies non-maximum suppression to filter detections.
            - It also ensures that detections are related to the original images considering padding.
            - Bounding boxes are normalized and reshaped as needed, the output shape of the boxes is [x1, y1, x2, y2].
            - Detections below a minimum size threshold are filtered.
            - Results are stitched together and organized for each image/frame.
            - If no valid detections are found, an empty list is returned for each image.

        """


        frames, boxes, scores, labels = non_max_suppression(preds, max_det=self.max_det, conf_thres_cls=self.conf_thrs)

        # remove detections that are not for the original images
        frames, boxes, scores, labels = self._filter(frames < n_ims, frames, boxes, scores, labels)

        # only run if we got boxes
        detections = None
        if boxes.numel() > 0:
            boxes = self._normalize_boxes(boxes)

            boxes = self._reshape_boxes(boxes, frames, orig_shapes) if orig_shapes else boxes

            frames, boxes, scores, labels = self._ensure_min_size(frames, boxes, scores, labels)

            detections = self._stitch_results(frames, boxes, scores, labels, n_ims)

        return detections if detections else [[] for _ in range(n_ims)]

    def _stitch_results(self, frames: torch.Tensor,
                        boxes: torch.Tensor,
                        scores: torch.Tensor,
                        labels: torch.Tensor,
                        n_ims: int) -> List[List[Union[List[float], bytes, float]]]:
        """
        Stitch together and organize detection results for multiple images.

        Args:
            frames (torch.Tensor): A tensor representing frames or images.
            boxes (torch.Tensor): A tensor containing bounding boxes in the format [x1, y1, x2, y2].
            scores (torch.Tensor): A tensor containing confidence scores.
            labels (torch.Tensor): A tensor containing class labels.
            n_ims (int): The number of images in the batch.

        Returns:
            List[List[Union[List[float], bytes, float]]]: A list of lists containing detection results for each image.
                Each inner list contains information for one detection, including the bounding box coordinates,
                class label (as a string encoded in UTF-8), and confidence score.
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
                res.append([box, str(self.ids[label]).encode("utf-8"), float(score)])
            results.append(res)
        return results

    def _ensure_min_size(self, frames, boxes, scores, labels) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Ensure that the normalized bounding boxes are at least a certain percentage of the image size.

        Args:
            frames (torch.Tensor): A tensor representing frames or images indices.
            boxes (torch.Tensor): A tensor containing bounding boxes in the format [x1, y1, x2, y2] normalized to [0, 1].
            scores (torch.Tensor): A tensor containing confidence scores for the boxes.
            labels (torch.Tensor): A tensor containing class labels for the boxes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple with tensors filtered to remove boxes below the minimum size threshold.
        """


        box_size_select = torch.logical_and(
            (boxes[:, 2] - boxes[:, 0]) > self.box_min_perc, # width
            (boxes[:, 3] - boxes[:, 1]) > self.box_min_perc, # height
        )

        return self._filter(box_size_select, frames, boxes, scores, labels)

    def _normalize_boxes(self, boxes: torch.Tensor) -> torch.Tensor:

        """
        Normalize bounding boxes to the range [0, 1].

        Args:
            boxes (torch.Tensor): A tensor containing bounding boxes in the format [x1, y1, x2, y2] in pixel coordinates.

        Returns:
            torch.Tensor: Bounding boxes normalized to the range [0, 1].
        """

        boxes /= self.imgsz
        # clamp boxes between 0 and 1
        boxes = boxes.clamp(0.0, 1.0)
        return boxes


    def _reshape_boxes(self, boxes: torch.Tensor, frames: torch.Tensor, orig_shapes: List[List[int]]) -> torch.Tensor:

        """
        Reshape bounding boxes to match the original image dimensions.

        Args:
            boxes (torch.Tensor): A tensor containing bounding boxes in the format [x1, y1, x2, y2] normalized to [0, 1].
            frames (torch.Tensor): A tensor representing frame or image indices.
            orig_shapes (List[List[int]]): A list of lists containing original image or frame shapes format: [H, W].

        Returns:
            torch.Tensor: Reshaped bounding boxes normalized to the original image dimensions in the format [x1, y1, x2, y2].
        """

        if orig_shapes is None:
            return boxes

        orig_shapes = torch.tensor(orig_shapes).to(frames.device)
        orig_shapes = orig_shapes[frames]

        resize_ratio = torch.min(self.imgsz / orig_shapes, dim=-1).values
        delta = (self.imgsz - (orig_shapes * resize_ratio.unsqueeze(1))) / 2
        abs_boxes = boxes * self.imgsz
        boxes[:, 0] = ((abs_boxes[:, 0] - delta[:, 1]) / resize_ratio) / orig_shapes[:, 1] # x1
        boxes[:, 1] = ((abs_boxes[:, 1] - delta[:, 0]) / resize_ratio) / orig_shapes[:, 0] # y1
        boxes[:, 2] = ((abs_boxes[:, 2] - delta[:, 1]) / resize_ratio) / orig_shapes[:, 1] # x2
        boxes[:, 3] = ((abs_boxes[:, 3] - delta[:, 0]) / resize_ratio) / orig_shapes[:, 0] # y2

        # clamp boxes between 0 and 1
        boxes = boxes.clamp(0.0, 1.0)
        return boxes


    def _images_same_shapes(self, ims: List[np.ndarray]) -> bool:
        """
        Check if all the images in a list have the same dimensions (imgsz, imgsz, 3)

        Args:
            ims (List[np.ndarray]): A list of NumPy arrays representing images.

        Returns:
            bool: True if all images have the same dimensions, False otherwise.
        """

        return all([im.shape == (self.imgsz, self.imgsz, 3) for im in ims])


if __name__ == '__main__':

    detector = FusionFaceLogoDetector(model_path="/home/ec2-user/dev/ctx-logoface-detector/artifacts/yolov8s_t74_best.torchscript",
                        conf_thrs={"logo": 0.8, "face": 0.1}, device="cuda")

    # read an image from path and convert it to numpy
    img_path = "/home/ec2-user/dev/data/logo05fusion/yolo/images/val/000000053.jpg"
    from PIL import Image
    img = Image.open(img_path)
    img = np.array(img)
    # print(img)
    img = [img for _ in range(3)]
    orig_shapes = [[400, 400], [800, 800], [600, 600]]
    detections = detector.detect(img, orig_shapes)
    pp.pprint(detections)
