import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from fusiondetector.yolov8.inference.nms import non_max_suppression
from fusiondetector import IMAGES_DIR, ARTIFACTS_DIR, OUTPUTS_DIR
# from compute_common.logger import logger
torch.set_printoptions(sci_mode=False)


class Resizer:
    """
    A simple Resizer.
    """

    def __init__(self, target_size: int, padding=True):
        self.target_shape = (target_size, target_size)
        self.padding = padding


    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize an image to a square image with padding to keep the same aspect ratio.
        """
        import cv2
        target_h, target_w = self.target_shape
        height, width, _ = image.shape
        scale = min(float(target_w) / float(width), float(target_h) / float(height))
        new_width, new_heigth = int(scale * width), int(scale * height)

        image = cv2.resize(image, (new_width, new_heigth))
        if self.padding:
            x_0 = int((target_h - image.shape[0]) / 2)
            x_1 = target_h - (image.shape[0] + x_0)
            y_0 = int((target_w - image.shape[1]) / 2)
            y_1 = target_w - (image.shape[1] + y_0)
            image = np.pad(image, [[x_0, x_1], [y_0, y_1], [0, 0]])

        if image.dtype == np.uint8:
            return image
        else:
            return image.astype(np.uint8)

class FusionDetector:

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
        conf_thrs: Union[dict, float, list] = None,
        box_min_perc: Union[dict, float, list] = 0.01,
        batch_size: int = -1,
        fp16: bool = True,
        device: str = "cuda:0",
        max_det: int = 100,
    ):
        jit = Path(model_path).suffix[1:] == "torchscript"
        if not jit:
            raise NotImplementedError(f'Model path suffix must be .torchscript got {model_path}')

        # set the device and fp16
        self.device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.fp16 = fp16 if self.device.type != 'cpu' else False

        # load the model and metadata
        self.model, self.metadata = self._load_jit(model_path, self.device, self.fp16)

        # read the metadata
        self.nc = self.metadata["nc"]
        self.ids = {int(k): v for k, v in self.metadata["ids"].items()} # convert keys to int
        self.names = {int(k): v for k, v in self.metadata["names"].items()} # convert keys to int
        self.imgsz = self.metadata["imgsz"][0]
        self.default_thrs = self.metadata["default_thrs"]

        print(f"Fusion Model - Classes: {self.nc}, Names: {self.names}, Image Size: {self.imgsz}, Ids: {self.ids}")

        # check confidence thresholds
        self.conf_thrs = self._check_thresholds(conf_thrs)

        # check the batch size  - batch size cannot be 0 or lower than -1
        self.batch_size = batch_size if batch_size > 0 or batch_size == -1 else 1

        self.box_min_percs = self._check_min_perc(box_min_perc)
        self.max_det = max_det

        print("Box Min Percs: ", self.box_min_percs)
        print("Confidence Thresholds: ", self.conf_thrs)

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

    def _validate_and_transform_input(self, input_data: Union[dict, list, float], default_data: Union[dict, list, float], input_name: str, bounds:List[float]):
        """
        Validates and transforms the input into a list of floats.

        Args:
            input_data (Union[dict, list, float]): Input data to be transformed into a list of floats.
            default_data (Union[dict, list, float]): Default data to be used if input_data is None.
            input_name (str): Name of the input, used for error messages.
            bounds (list): List of floats representing the allowable bounds for the input data.

        Returns:
            list: List of floats in order of classes.
        """

        # checks

        # Ensure that bounds is of length 2 and that bound[0] < bound[1]
        assert len(bounds) == 2, f"Bounds must be a list of length 2, got {len(bounds)}"
        assert bounds[0] < bounds[1], f"Bounds must be in the format [lower, upper], got {bounds}"

        # Ensure that at least default_data is present
        assert default_data is not None, f"Default data of {input_name} must be provided"

        # check input_data is either a float, list, or dictionary or None
        assert input_data is None or isinstance(input_data, (float, list, dict)), f"{input_name} must be a float, list, or dictionary, got {type(input_data)}"

        # checks done

        output = input_data if input_data is not None else default_data

        # If the input is a float, create a list of floats with the same value
        if isinstance(output, float):
            output = [output] * self.nc
        # If it's a list, check if it's valid and has the expected length
        elif isinstance(output, list):
            if not len(output) == self.nc:
                raise ValueError(f"{input_name} list must be of length {self.nc}, got {len(output)}")
        # If it's a dictionary, process it based on specific rules
        elif isinstance(output, dict):
            # Get the names from self.names and order them by their keys (0, 1, ...)
            names = [str(self.names[k]).lower() for k in sorted(self.names.keys())]

            # Create two lists from the ids: one with the ids as strings and one as ints, sorted by the keys
            ids_str = [str(self.ids[k]) for k in sorted(self.ids.keys())]
            ids_int = [int(id_) for id_ in ids_str]

            # Case when names are passed
            if set(output.keys()) == set(names):
                output = [output[n] for n in names]
            # Case when ids are passed
            elif set(output.keys()) in [set(ids_str), set(ids_int)]:
                output = {int(k): v for k, v in output.items()}  # Convert keys to int
                output = [output[k] for k in ids_int]
            else:
                raise NotImplementedError(f"{input_name} keys must be: {names}, {ids_int} got {set(output.keys())}")

        # Ensure that the values are within the specified bounds
        if not all([bounds[0] <= t <= bounds[1] for t in output]):
            raise ValueError(f"{input_name} must be between {bounds[0]} and {bounds[1]}, got {output}")

        return output


    def _check_thresholds(self, conf_thrs: Union[dict, float, list]) -> List[float]:
        """
        Check and transform confidence thresholds.

        Args:
            conf_thrs (Union[dict, float, list]): Confidence thresholds can have 3 formats:
                1. None
                2. float
                3. Dictionary with keys as ids or names and values as floats.
        Returns:
            List[float]: List of confidence thresholds values in the order 0,1,2...
        Raises:
            NotImplementedError: If the input format is not supported.
        """

        return self._validate_and_transform_input(conf_thrs, self.default_thrs, "Confidence thresholds", [0.0, 1.0])


    def _check_min_perc(self, box_min_perc: Union[dict, float, list]) -> List[float]:

        return self._validate_and_transform_input(box_min_perc, 0.0, "Box min percentage", [0.0, 1.0])


    def warmup(self) -> None:
        """
        Warm up the detector model.

        This method prepares the detector model for inference by generating random input images and running
        them through the detector. It ensures that the model is loaded and initialized properly.

        Returns:
            None
        """

        # logger.info("Warming up the detector model...")
        print("Warming up the detector model...")
        im = np.array(np.random.uniform(0, 255, [self.imgsz, self.imgsz, 3]), dtype=np.float32)

        im = [im for _ in range(self.batch_size)] if self.batch_size > 1 else [im]

        self.detect(im)
        # logger.info("Warmup finished.")
        print("Warmup finished.")

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
        assert self._images_same_shapes(ims), f"Expected input images to have the same shape, ({self.imgsz}, {self.imgsz}, 3)."

        n_ims = len(ims)
        batch_size = self.batch_size if self.batch_size != -1 else n_ims

        assert batch_size >= n_ims, f"Batch size must be greater than or equal to number of images or set as -1 for dynamic batch size."

        if batch_size > n_ims:
            n_ims_add = batch_size - n_ims
            black_im = np.zeros((self.imgsz, self.imgsz, 3)).astype(np.uint8) # HWC
            # black_im = (black_im * 255).astype(np.uint8)
            ims += [black_im for _ in range(n_ims_add)] # [B x [HWC]]


        # convert to torch tensor
        ims = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).contiguous().float() # BCHW
        ims = ims.to(self.device)
        ims = ims.half() if self.fp16 else ims.float() # uint8 to fp16/32
        ims /= 255.0 # 0 - 255 to 0.0 - 1.0
        # check if  we indeed are  using the right dtype
        assert ims.dtype == torch.float16 if self.fp16 else torch.float32

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

    def _ensure_min_size(self,
                        frames:torch.Tensor,
                        boxes:torch.Tensor,
                        scores:torch.Tensor,
                        labels:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

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

        assert len(self.box_min_percs) == self.nc, f"Box min percentage must be a list of length {self.nc} got {len(self.box_min_percs)}"

        box_min_percs = torch.tensor(self.box_min_percs).to(boxes.device)

        box_size_select = torch.logical_and(
            (boxes[:, 2] - boxes[:, 0]) > box_min_percs[labels], # width
            (boxes[:, 3] - boxes[:, 1]) > box_min_percs[labels], # height
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

    import os
    from PIL import Image
    import numpy as np
    import pprint as pp
    from fusiondetector.utils.boxes import plot_boxes


    bs = 3
    img_path = os.path.join(IMAGES_DIR, "tiktok.png")
    # model_path = os.path.join(ARTIFACTS_DIR, "yolov8m_t0_epoch4.torchscript")
    # model_path = "/home/ec2-user/dev/yolo/runs/detect/train7/weights/epoch5.torchscript"
    # model_path = "/home/ec2-user/dev/yolo/runs/detect/train8/weights/epoch55.torchscript"
    # model_path = "/home/ec2-user/dev/yolo/runs/detect/train11/weights/epoch14.torchscript"
    # model_path = "/home/ec2-user/dev/yolo/runs/detect/train11/weights/last.torchscript"
    model_path = "/home/ec2-user/dev/yolo/runs/detect/train14/weights/best.torchscript"
    device = "cpu"
    imgsz = 416

    # Import and initialize the FusionFaceLogoDetector class with the specified model path and device.
    detector = FusionDetector(model_path = model_path,
                            device = device,
                            batch_size = bs,
                            conf_thrs=0.15,
                            box_min_perc = {'logo': 0.01, 'face': 0.01})

    # Open the input image, convert it to RGB, and convert it to a numpy array.
    img = np.array(Image.open(img_path).convert("RGB"))

    # Store a copy of the original image and its dimensions.
    orig_img, orig_w, orig_h = img, img.shape[1], img.shape[0]
    # Store the original shape of the image.
    orig_shape = [orig_h, orig_w]

    img = Resizer(target_size=imgsz).resize(img)

    # Create a black image with dimensions 416x416x3.
    black_img = np.zeros((imgsz, imgsz, 3)).astype(np.uint8)

    # Create a list containing black and original images.
    img = [black_img] * (bs-1) + [img]

    # Create a list containing the original shapes
    orig_shapes = [[imgsz, imgsz] for _ in range(bs-1)] + [orig_shape]

    # Use the detector to perform object detection on the list of images.
    detections = detector.detect(img, orig_shapes)
    pp.pprint(detections)

    # Extract the last image (the original size image) and its detected boxes.
    img, boxes = img[-1], detections[-1]

    # Create a modified list of boxes, marking boxes with label '90020' as '1' and others as '0'.
    only_boxes = [[1 if box[1] == b'90020' else 0] + box[0] for box in boxes]

    # Extract the scores of the detected boxes.
    scores = [box[2] for box in boxes]

    # Plot the boxes on the original image, saving the result as an image file.
    plot_boxes(orig_img, boxes=only_boxes, box_type="xyxy", save=True, scores=scores, output_dir = OUTPUTS_DIR)
