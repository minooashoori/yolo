
def transform_xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h

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

def transform_xyxy_to_yolo(box):
    x1, y1, x2, y2 = box
    x_center = (x1 + x2)/2
    y_center = (y1 + y2)/2
    w = x2 - x1
    h = y2 - y1
    return x_center, y_center, w, h



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


# test intersection_area_yolo, union_area_yolo, iou_yolo
box1 = [0.75, 0.75, 0.5, 0.5]
box2 = [0.5, 0.5, 0.5, 0.5]
iou = iou_yolo(box1, box2)
print(iou)