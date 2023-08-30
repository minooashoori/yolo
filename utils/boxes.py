import ast
import seaborn as sns
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches

def transform_xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h

def transform_xywh_to_xyxy(box):
    x, y, w, h = box
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return x1, y1, x2, y2

def transform_xywh_to_yolo(box):
    x, y, w, h = box
    x_center = x + w/2
    y_center = y + h/2
    return x_center, y_center, w, h

def transform_yolo_to_xyxy(box):
    x_center, y_center, w, h = box
    x1 = x_center - w/2
    y1 = y_center - h/2
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2

def transform_yolo_to_xywh(box):
    x_center, y_center, w, h = box
    x1 = x_center - w/2
    y1 = y_center - h/2
    return x1, y1, w, h

def transform_xyxy_to_yolo(box):
    x1, y1, x2, y2 = box
    x_center = (x1 + x2)/2
    y_center = (y1 + y2)/2
    w = x2 - x1
    h = y2 - y1
    return x_center, y_center, w, h

def is_percentage(box):
    # return all(0 <= coord <= 1 for coord in box)
    mean = sum(box)/len(box)
    return  0 <= mean <= 1

def fix_bounds_xyxy_box(box, image_size, is_relative):
    x1, y1, x2, y2 = box
    # check if box is outside bounds
    if is_relative:
        if x1 < 0: # x1
            x1 = 0.0
        if y1 < 0: # y1
            y1 = 0.0
        if x2 > 1.0: # x2
            x2 = 1.0
        if y2 > 1.0: # y2
            y2 = 1.0
    else:
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > image_size[0]:
            x2 = image_size[0]
        if y2 > image_size[1]:
            y2 = image_size[1]

    return x1, y1, x2, y2


def transform_box(box, image_size, box_type, is_relative):

    assert box_type in ["yolo", "xywh", "xyxy"], "box_type must be one of: 'yolo', 'xywh' or 'xyxy'"


    w, h = image_size
    x1, y1, x2, y2 = box

    if is_relative:
        if not is_percentage(box):
            # we will convert the absolute coordinates to relative coordinates
            x1 = x1/w
            x2 = x2/w
            y1 = y1/h
            y2 = y2/h
    else:
        if is_percentage(box):
            # we will convert the relative coordinates to absolute coordinates
            x1 = int(x1*w)
            x2 = int(x2*w)
            y1 = int(y1*h)
            y2 = int(y2*h)

    box = fix_bounds_xyxy_box(box=[x1, y1, x2, y2], image_size=image_size, is_relative=is_relative)

    if box_type == 'yolo':
        box = transform_xyxy_to_yolo(box)
    elif box_type == 'xywh':
        box = transform_xyxy_to_xywh(box)

    return box



def intersection_area_yolo(box1, box2):
    x_center1, y_center1, w1, h1 = box1
    x1 = x_center1 - w1 / 2
    y1 = y_center1 - h1 / 2
    x2 = x1 + w1
    y2 = y1 + h1

    x_center2, y_center2, w2, h2 = box2
    _x1 = x_center2 - w2 / 2
    _y1 = y_center2 - h2 / 2
    _x2 = _x1 + w2
    _y2 = _y1 + h2

    xA = max(x1, _x1)
    yA = max(y1, _y1)
    xB = min(x2, _x2)
    yB = min(y2, _y2)

    width = max(0, xB - xA)
    height = max(0, yB - yA)

    area = width * height

    return area


def union_area_yolo(box1, box2):
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    return area1 + area2 - intersection_area_yolo(box1, box2)

def iou_yolo(box1, box2):
    return intersection_area_yolo(box1, box2) / union_area_yolo(box1, box2)


def round_box(box):
    return [round(coord, 4) for coord in box]


def transf_any_box(box, input_type="xyxy", output_type="yolo"):


    if input_type not in ["xyxy", "xywh", "yolo"]:
        raise ValueError(f"input_type: {input_type} is not supported")
    if output_type not in ["xyxy", "xywh", "yolo"]:
        raise ValueError(f"output_type: {output_type} is not supported")

    if input_type == output_type:
        pass # do nothing

    # xyxy -> ...
    if input_type == "xyxy" and output_type == "yolo":
        box = transform_xyxy_to_yolo(box)


    if input_type == "xyxy" and output_type == "xywh":
        box = transform_xyxy_to_xywh(box)

    # xywh -> ...
    if input_type == "xywh" and output_type == "xyxy":
        box = transform_xywh_to_xyxy(box)


    if input_type == "xywh" and output_type == "yolo":
        box = transform_xywh_to_yolo(box)

    # yolo -> ...
    if input_type == "yolo" and output_type == "xyxy":
        box = transform_yolo_to_xyxy(box)

    if input_type == "yolo" and output_type == "xywh":
        box = transform_yolo_to_xywh(box)

    box = round_box(box)
    return box



def relative(width, height, box):
    if not is_percentage(box):
        box = box[0]/width, box[1]/height, box[2]/width, box[3]/height
    return [round(coord, 4) for coord in box]

def ensure_bounds(coord, min_coord, max_coord):
    coord = max(coord, min_coord)
    coord = min(coord, max_coord)
    coord = round(coord, 4)
    return coord

def ensure_order(box, box_type="xyxy"):
    if box_type != "xyxy":
        raise ValueError(f"box_type: {box_type} is not supported")
    x1, y1, x2, y2 = box

    if x1 >= x2:
        # swap x1 and x2
        x1, x2 = x2, x1
    if y1 >= y2:
        # swap y1 and y2
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def fix_bounds_relative(box, box_type):

    new_box = []
    for coord in box:
        new_box.append(ensure_bounds(coord, 0.0, 1.0))

    if box_type == "xyxy":
        new_box = ensure_order(new_box, box_type)

    return new_box

def yolo_annotations(boxes):
    if len(boxes) == 0:
        return ""
    content = ""
    for box in boxes:
        category = box.category
        x, y, w, h = box.x, box.y, box.width, box.height
        x = ensure_bounds(x, 0.001, 0.999)
        y = ensure_bounds(y, 0.001, 0.999)
        w = ensure_bounds(w, 0.001, 0.999)
        h = ensure_bounds(h, 0.001, 0.999)
        
        line = f"{category} {x} {y} {w} {h}\n"
        content += line
    return content

def make_patch_format(yolo_box, width, height):
    x_c, y_c, w, h = yolo_box
    x1 = x_c - w/2
    y1 = y_c - h/2
    x1 = x1 * width
    y1 = y1 * height
    w = w * width
    h = h * height
    return x1, y1, w, h


def get_boxes_from_yolo_annotation(annotation):
    # open the annotation file
    # check if it's a file or a string
    if os.path.isfile(annotation):
        with open(annotation, 'r') as f:
            content = f.read()
    else:
        #if it's a string
        content = annotation

    # split the content by new line
    lines = content.split("\n")
    # remove the last line if it is empty
    if lines[-1] == "":
        lines = lines[:-1]
    # split each line by space
    lines = [line.split(" ") for line in lines]
    # convert each line to a list of  int and floats
    lines = [[int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])] for line in lines]
    return lines


def plot_boxes(path_or_img, boxes=None, yolo_annotation=None, save=False):

    if boxes is None and yolo_annotation is None:
        raise ValueError("Either boxes or yolo_annotation must be provided.")
    if boxes is not None and yolo_annotation is not None:
        raise ValueError("Only one of boxes or yolo_annotation must be provided.")


    if yolo_annotation is not None:
        boxes = get_boxes_from_yolo_annotation(yolo_annotation)
        # get the number of categories
        categories = [box[0] for box in boxes]
        categories = set(categories)
        n_categories = len(categories)
    else:
        # add a fictitious category to each box
        boxes = [[0] + box for box in boxes]
        categories = [0]
        n_categories = 1

    # if path_or_img is a path, we need to open the image
    if isinstance(path_or_img, str):
        img = Image.open(path_or_img)
    # if it's numpy array, we need to convert it to PIL image
    elif isinstance(path_or_img, np.ndarray):
        # read the numpy array as a PIL image
        # check if it's in the range 0-1 or 0-255
        if path_or_img.max() <= 1.0:        
            path_or_img = path_or_img * 255

        path_or_img = path_or_img.astype(np.uint8)
        img = Image.fromarray(path_or_img)
    elif isinstance(path_or_img, Image.Image):
        img = path_or_img

    width, height = img.size

    # create colors for each category
    colors_list = sns.color_palette("hls", n_categories)
    # create a map from category to color
    colors = {category: color for category, color in zip(categories, colors_list)}
    # we need to make the figure and the axes
    fig, ax = plt.subplots(1)

    ax.imshow(img)
    if len(boxes) > 0:
        for box_ in boxes:
            category, box = box_[0], box_[1:]
            x1, y1, w, h = make_patch_format(box, width, height)
            color = colors[category]
            # create a rectangle patch
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
            # add the patch to the axes
            ax.add_patch(rect)
    if save:
        plt.savefig("boxes.jpg", bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    # test intersection_area_yolo, union_area_yolo, iou_yolo
    box1 = [0.75, 0.75, 0.5, 0.5]
    box2 = [0.5, 0.5, 0.5, 0.5]
    iou = iou_yolo(box1, box2)
    print(iou)