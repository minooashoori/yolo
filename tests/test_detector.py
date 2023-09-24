import unittest
import numpy as np
from fusiondetector.yolov8.inference.detector import FusionFaceLogoDetector  # Assuming your class is in a file named fusion_face_logo_detector.py
import torch

class TestFusionFaceLogoDetector(unittest.TestCase):

    real_model_path = "/home/ec2-user/dev/ctx-logoface-detector/artifacts/yolov8m_t0_epoch4.torchscript"

    def test_invalid_model_path(self):
        # Test initializing the detector with an invalid model path
        model_paths = [
            "",
            "invalid_model.onnx",
            "invalid_model.pt",
            "/path/to/nothing",
        ]
        for model_path in model_paths:
            with self.subTest(model_path=model_path):
                with self.assertRaises(NotImplementedError):
                    FusionFaceLogoDetector(model_path=model_path)

    def test_empty_model_path(self):
        # Test initializing the detector with an empty model path
        with self.assertRaises(TypeError):
            FusionFaceLogoDetector(model_path=None)

    def test_valid_model_paths(self):
        # Test initializing the detector with valid model paths
        model_paths = [
            self.real_model_path,
        ]
        for model_path in model_paths:
            with self.subTest(model_path=model_path):
                detector = FusionFaceLogoDetector(model_path=model_path)
                self.assertIsNotNone(detector)


    def test_invalid_confidence_thresholds(self):
        # Test initializing the detector with invalid confidence thresholds
        with self.assertRaises(NotImplementedError):
            FusionFaceLogoDetector("valid_model.torchscript", conf_thrs={"facesss": 0.5, "logos": 0.6})
        with self.assertRaises(NotImplementedError):
            FusionFaceLogoDetector("valid_model.torchscript", conf_thrs={"face": 0.5, "logo": 0.6, "invalid": 1.2})
        with self.assertRaises(ValueError):
            FusionFaceLogoDetector("valid_model.torchscript", conf_thrs={"face": 1.5, "logo": 0.6})

    def test_initialize_on_cpu(self):
        # Test initializing the detector on CPU
        detector = FusionFaceLogoDetector(
            model_path=self.real_model_path,
            conf_thrs={"face": 0.5, "logo": 0.6},
            device="cpu"  # Initialize on CPU
        )

        # Ensure that the detector's device is CPU
        self.assertEqual(detector.device, torch.device("cpu"))
        # ensure we are runnning in float mode
        self.assertFalse(detector.fp16)

    def setUp(self):
        # Initialize the detector with a sample TorchScript model (provide the actual path)
        self.detector = FusionFaceLogoDetector(model_path=self.real_model_path)

    def test_initialization(self):
        # Test the initialization of the detector
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.batch_size, -1)
        self.assertEqual(self.detector.device, torch.device('cuda:0'))
        self.assertTrue(self.detector.fp16)
        self.assertEqual(self.detector.max_det, 100)

    def test_warmup(self):
        # Test if the warmup method runs without errors
        self.detector.warmup()

    def test_images_same_shapes(self):
        # Test _images_same_shapes method
        image_list = [
            np.random.rand(416, 416, 3).astype(np.float32),
            np.random.rand(416, 416, 3).astype(np.float32),
            np.random.rand(416, 416, 3).astype(np.float32),
        ]
        self.assertTrue(self.detector._images_same_shapes(image_list))

    def test_images_same_shapes_different_shapes(self):
        # Test _images_same_shapes method with images of diffent sizes
        image_list = [
            np.random.rand(224, 224, 3).astype(np.float32),
            np.random.rand(224, 224, 3).astype(np.float32),
            np.random.rand(500, 224, 3).astype(np.float32),
            np.random.rand(224, 224, 3).astype(np.float32),
        ]
        self.assertFalse(self.detector._images_same_shapes(image_list))

    def test_preprocess(self):
        # Test the preprocessing method on a sample image (provide a sample image)
        sample_image = np.random.uniform(0, 255, [self.detector.imgsz, self.detector.imgsz, 3]).astype(np.float32)
        ims, n_ims = self.detector._preprocess([sample_image])

        # Ensure ims is a torch.Tensor and n_ims is an integer
        self.assertIsInstance(ims, torch.Tensor)
        self.assertIsInstance(n_ims, int)


    def test_normalize_boxes(self):
        # Test _normalize_boxes method
        boxes = torch.tensor([[10, 20, 100, 200], [20, 30, 150, 250]]).float()
        normalized_boxes = self.detector._normalize_boxes(boxes)
        self.assertTrue(torch.all(normalized_boxes >= 0) and torch.all(normalized_boxes <= 1))


    def test_detect_empty_input(self):
        # Test object detection on an empty list of images
        detections = self.detector.detect([])
        self.assertEqual(len(detections), 0)

    def test_detect(self):
        # Test object detection on a sample image (provide a sample image)
        sample_image = np.random.uniform(0, 255, [self.detector.imgsz, self.detector.imgsz, 3]).astype(np.float32)
        detections = self.detector.detect([sample_image])

        # Ensure that detections is a list
        self.assertIsInstance(detections, list)

        # Ensure that detections for the sample image are also a list
        self.assertIsInstance(detections[0], list)

        # Ensure that detections contain the expected information (bounding boxes, labels, scores)
        for detection in detections[0]:
            self.assertIsInstance(detection, list)
            self.assertEqual(len(detection), 3)  # [bounding box, label, score]
            self.assertIsInstance(detection[0], list)  # Bounding box
            self.assertIsInstance(detection[1], bytes)  # Label
            self.assertIsInstance(detection[2], float)  # Score

    def test_detect_batch(self):
        # Test object detection on a batch of sample images (provide a batch of sample images)
        sample_image = np.random.uniform(0, 255, [self.detector.imgsz, self.detector.imgsz, 3]).astype(np.float32)
        detections = self.detector.detect([sample_image, sample_image, sample_image])

        # Ensure that detections is a list
        self.assertIsInstance(detections, list)

        # Ensure that detections for the sample image are also a list
        self.assertIsInstance(detections[0], list)

        # Ensure that detections contain the expected information (bounding boxes, labels, scores)
        for detection in detections[0]:
            self.assertIsInstance(detection, list)
            self.assertEqual(len(detection), 3)

    def test_detect_cpu(self):
        # set the model to CPU
        detector = FusionFaceLogoDetector(model_path=self.real_model_path, device="cpu")
        sample_image = np.random.uniform(0, 255, [self.detector.imgsz, self.detector.imgsz, 3]).astype(np.float32)
        detections = detector.detect([sample_image])

                # Ensure that detections is a list
        self.assertIsInstance(detections, list)

        # Ensure that detections for the sample image are also a list
        self.assertIsInstance(detections[0], list)

        # Ensure that detections contain the expected information (bounding boxes, labels, scores)
        for detection in detections[0]:
            self.assertIsInstance(detection, list)
            self.assertEqual(len(detection), 3)  # [bounding box, label, score]
            self.assertIsInstance(detection[0], list)  # Bounding box
            self.assertIsInstance(detection[1], bytes)  # Label
            self.assertIsInstance(detection[2], float)  # Score





if __name__ == '__main__':
    unittest.main()
