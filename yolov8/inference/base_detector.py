"""
Detector module.
"""
from compute_engine.misc.lazy_importer import LazyImport

with LazyImport():
    import numpy as np
    import tensorflow as tf

from typing import List, Tuple
from compute_common.logger import logger



class Detector:
    """
    Detector class.
    Loads a deep learning model and uses it to detect items in images.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = None,
        detection_min_size_percentage: int = 0.1,
        batch_size: int = None,
        process_gpu_memory_fraction: float = 1.0
    ):
        self.model_path = model_path
        self.batch_size = batch_size

        tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True  # pylint: disable=no-member
        tf_config.gpu_options.per_process_gpu_memory_fraction = process_gpu_memory_fraction  # pylint: disable=no-member

        self.tf_session = tf.compat.v1.Session(config=tf_config)
        self.input_image_size = 416  # CONSTANT
        self.confidence_threshold = confidence_threshold
        self.detection_min_size_percentage = detection_min_size_percentage

        with self.tf_session.as_default():  # pylint: disable=not-context-manager
            protobuf = tf.compat.v1.saved_model.loader.load(
                self.tf_session, [
                    tf.compat.v1.saved_model.SERVING], self.model_path
            )
            if "images" in protobuf.signature_def["predict_images"].inputs:
                self.input_placeholder = protobuf.signature_def["predict_images"].inputs["images"].name
            else:
                self.input_placeholder = protobuf.signature_def["predict_images"].inputs["img_placeholder"].name
            self.box_coordonates = (
                protobuf.signature_def["predict_images"].outputs["bbox_coor"].name
            )
            self.box_scores = (
                protobuf.signature_def["predict_images"].outputs["score"].name)
            self.box_labels = (
                protobuf.signature_def["predict_images"].outputs["label"].name)
            self.frame_nums = (
                protobuf.signature_def["predict_images"].outputs["frame_num"].name)

            if "conf_loss_pred" in protobuf.signature_def["predict_images"].outputs:
                self.active_learning_scores = (
                    protobuf.signature_def["predict_images"].outputs["conf_loss_pred"].name
                )
                self.features = (
                    protobuf.signature_def["predict_images"].outputs["conf_feats_concat"].name
                )
            elif "active_learning_loss" in protobuf.signature_def["predict_images"].outputs:
                self.active_learning_scores = (
                    protobuf.signature_def["predict_images"].outputs["active_learning_loss"].name
                )
                self.features = (
                    protobuf.signature_def["predict_images"].outputs["active_learning_features"].
                    name
                )
            else:
                self.active_learning_scores = None
                self.features = None

        self.warm_up()

    def warm_up(self):
        logger.info("warm_up detector ...")
        img_size = (self.input_image_size, self.input_image_size, 3)
        batch_size = 1 if self.batch_size is None else self.batch_size
        frames = np.random.rand(batch_size, *img_size)
        self.detect(frames, batch_size*[[512, 512]])
        logger.info("warm_up detector done")

    
    def detect(self, images: List[np.ndarray], original_shape):
        """
        Runs a batched detection on images, and resizes detections if original_shape is given.
        All the images MUST have the same shape.
        """
        assert self._are_images_same_shape(images)

        number_input_images = len(images)
        if self.batch_size is not None:
            missing_img = self.batch_size - len(images)
            black_img = np.zeros(
                (self.input_image_size, self.input_image_size, 3))
            for _ in range(missing_img):
                images.append(black_img)

        frame_nums, detections, scores, labels, al_scores, features = self._predict_detections(
            images
        )

        boxes_labels_scores = self._postprocessing(
            frame_nums, detections, scores, labels, original_shape, number_input_images
        )

        return boxes_labels_scores, al_scores, features

    
    def _postprocessing(
        self, frame_nums, detections, scores, labels, original_shape, number_input_images
    ):
        frame_nums = frame_nums.astype(np.int8)
        frame_nums, detections, scores, labels = self._filter(
            frame_nums < number_input_images, frame_nums, detections, scores, labels)

        if self.confidence_threshold:
            frame_nums, detections, scores, labels = self._filter_with_threshold(
                frame_nums,
                detections,
                scores,
                labels,
            )

        if original_shape:
            detections = self._transform_with_shape(
                detections, frame_nums, original_shape)
        
        detections = self._truncate_bounding_boxes(detections)

        frame_nums, detections, scores, labels = self._filter_with_min_size(
            frame_nums,
            detections,
            scores,
            labels,
        )

        boxes_labels_scores = []
        for i in range(number_input_images):
            p_select = np.equal(frame_nums, i)
            temp = []
            for detection, label, score in zip(
                detections[p_select].tolist(
                ), labels[p_select], scores[p_select]
            ):
                temp.append([detection, label.decode("utf-8"), score])
            boxes_labels_scores.append(temp)

        return boxes_labels_scores

    def _are_images_same_shape(self, images: List[np.ndarray]) -> bool:
        """
        Check if all images have the same shape.
        We only check for the two first channels: height and width.
        """
        return len({image.shape[:2] for image in images}) == 1

    def _predict_detections(
        self,
        images: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs a detection on images at one time.
        detections coordinates are between 0 and 1.
        Note: we transform detections by assuming the model returns
        detections between 0 and self.input_image_size

        Detections follows [x1, y1, x2, y2], in this landmark:
        - [0,0] is the top left corner of the image.
        - y axis is the height axis
        - x axis is the width axis.
        - (x1,y1) is the top left corner of the detection.
        - (x2,y2) is the bottom right corner of the detection.
        """
        if self.active_learning_scores is None:
            frame_nums, detections, scores, labels = self.tf_session.run(
                [
                    self.frame_nums,
                    self.box_coordonates,
                    self.box_scores,
                    self.box_labels,
                ],
                feed_dict={self.input_placeholder: np.stack(images)},
            )
            al_scores = np.zeros(len(images))
            features = [None] * len(images)
        else:
            frame_nums, detections, scores, labels, al_scores, features = self.tf_session.run(
                [
                    self.frame_nums, self.box_coordonates, self.box_scores, self.box_labels,
                    self.active_learning_scores, self.features
                ],
                feed_dict={self.input_placeholder: np.stack(images)},
            )

        detections = detections / self.input_image_size
        return frame_nums, detections, scores, labels, al_scores, features

    def _filter(
        self,
        selected: np.ndarray,
        frame_nums: np.ndarray,
        detections: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frame_nums = frame_nums[selected]
        detections = detections[selected]
        scores = scores[selected]
        labels = labels[selected]
        return frame_nums, detections, scores, labels

    def _filter_with_threshold(
        self,
        frame_nums: np.ndarray,
        detections: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter detections with the confidence threshold.
        """
        score_select = np.greater(scores, self.confidence_threshold)
        return self._filter(score_select, frame_nums, detections, scores, labels)

    def _filter_with_min_size(
        self,
        frame_nums: np.ndarray,
        detections: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter detections with a minimum size, expressed in percentage.
        """
        box_size_select = np.logical_and(
            np.greater(detections[:, 2] - detections[:, 0],
                       self.detection_min_size_percentage),
            np.greater(detections[:, 3] - detections[:, 1],
                       self.detection_min_size_percentage)
        )
        return self._filter(box_size_select, frame_nums, detections, scores, labels)

    def _truncate_bounding_boxes(self, detections: np.ndarray) -> np.ndarray:
        """
        Truncate detections between 0 and 1
        to avoid having detections partially outside the image.
        """
        detections[:, 0][detections[:, 0] < 0] = 0
        detections[:, 1][detections[:, 1] < 0] = 0
        detections[:, 2][detections[:, 2] > 1] = 1
        detections[:, 3][detections[:, 3] > 1] = 1
        return detections

    def _transform_with_shape(self, detections: np.ndarray, frame_nums: np.ndarray, original_shape):
        """
        Transforms the detections according to the frame original shape,
        and returns detections between [0,1].
        """
        original_shape = np.array(original_shape)
        original_shape = original_shape[frame_nums, :]
        resize_ratio = np.min(self.input_image_size/original_shape, axis=-1)
        delta = (self.input_image_size -
                 (original_shape * resize_ratio[:, np.newaxis]))/2
        temo_detections = detections*self.input_image_size
        detections[:, 0] = (
            (temo_detections[:, 0] - delta[:, 1]) / resize_ratio) / original_shape[:, 1]
        detections[:, 1] = (
            (temo_detections[:, 1] - delta[:, 0]) / resize_ratio) / original_shape[:, 0]
        detections[:, 2] = (
            (temo_detections[:, 2] - delta[:, 1]) / resize_ratio) / original_shape[:, 1]
        detections[:, 3] = (
            (temo_detections[:, 3] - delta[:, 0]) / resize_ratio) / original_shape[:, 0]
        return detections