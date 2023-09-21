import numpy as np
import torch
import torchvision


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y




def non_max_suppression(
        prediction,
        conf_thres=0.0,
        iou_thres=0.45,
        max_det=100,
        max_nms=30000,
        max_wh=7680,
        conf_thres_classes=None,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        max_det (int): The maximum number of boxes to keep after NMS.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels
        conf_thres_classes (List[float]): A list of confidence thresholds for each class. If specified, the length must equal the number of classes.


    Returns:
        # (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
        #     shape (num_boxes, 6) containing the kept boxes, with columns
        #     (x1, y1, x2, y2, confidence, class).
        frames (torch.Tensor): A tensor with the shape (B x N) containing the frame number for each detection.
        boxes (torch.Tensor): A tensor with the shape (B x N, 4) containing the coordinates of the kept boxes.
        scores (torch.Tensor): A tensor with the shape (B x N) containing the scores of the predicted labels.
        labels (torch.Tensor): A tensor with the shape (B x N) containing the predicted labels.
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    bs = prediction.shape[0]  # batch size
    nc = (prediction.shape[1] - 4)  # number of classes
    if conf_thres_classes:
        assert nc == len(conf_thres_classes), f'Number of classes {nc} does not match length of conf_thres_classes {len(conf_thres_classes)}'
        assert min(conf_thres_classes) >= 0, f'Invalid Confidence threshold {conf_thres_classes}, valid values are between 0.0 and 1.0'
        assert max(conf_thres_classes) <= 1, f'Invalid Confidence threshold {conf_thres_classes}, valid values are between 0.0 and 1.0'
        conf_thres = float(min(conf_thres_classes))
        conf_thres_classes = torch.tensor(conf_thres_classes, device=prediction.device)  # to gpu
    # nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates with one of the confs > conf_thres

    prediction = prediction.transpose(-1, -2)  # shape(B,6,6300) to shape(B,6300,6), i.e., box and confs in the last dimension
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # yolo to xyxy

    output = [torch.zeros((0, 7), device=prediction.device)] * bs # list of size bs, each el in shape (0,6)

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height

        x = x[xc[xi]]  # confidence, xc[xi] is a mask to select boxes with confidence > conf_thres (n_boxes, 6)


        # If no box remains process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls = x.split((4, nc), 1) # split the tensor in 2 one for  box and one for cls

        # best class only
        conf, j = cls.max(1, keepdim=True)
        if conf_thres_classes is not None:
            threshold_mask = conf >= conf_thres_classes[j]
        else:
            threshold_mask = conf > conf_thres

        x = torch.cat((box, conf, j.float()), 1)[threshold_mask.squeeze(-1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        if xi == 1:
            continue # TODO: remove this line
        
        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes this is will offset the boxes so they do not overlap
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS w/ IoU threshold
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        frame = torch.full_like(output[xi][:, 0], xi) # frame number
        #concat frame number to output
        output[xi] = torch.cat((frame.unsqueeze(1), output[xi]), 1)

    # print(output)



    output = torch.cat(output, 0)
    # print("---------")
    # print(output)
    #frames is the first column of output
    frames = output[:, 0].long() # make it a long tensor
    boxes = output[:, 1:5]
    scores = output[:, 5]
    labels = output[:, 6].long()

    return frames, boxes, scores, labels