from random import shuffle
from utils.io import list_files, mv_to_dir, compress_dir, unzip_file
import xml.etree.ElementTree as ET
import pandas as pd
import os
from databricks.preprocess.boxes import transf_any_box, relative, fix_bounds_relative
import shutil
# 

def process_xml(file):
    
    """
    this function will process a single xml file and return image size and bounding boxes
    """
    
    tree =  ET.parse(file)
    
    width = int(tree.find("size/width").text)
    height = int(tree.find("size/height").text)
    
    boxes = []
    for obj_elem in tree.findall(".//object"):
        xmin = int(obj_elem.find("bndbox/xmin").text)
        ymin = int(obj_elem.find("bndbox/ymin").text)
        xmax = int(obj_elem.find("bndbox/xmax").text)
        ymax = int(obj_elem.find("bndbox/ymax").text)
        
        boxes.append([xmin, ymin, xmax, ymax])
    
    image_size = (width, height)
    return image_size, boxes

def process_yolo_box(boxes):
    # boxes = [[xmin, ymin, xmax, ymax], ...]
    """
    this function will process a list of yolo boxes and return a file with the following format:
    class x_center y_center width height
    since there are only logos, all classes will be 1
    """
    
    lines = []  # List to store lines
    
    for box in boxes:
        x_center, y_center, w, h = box
        line = f"{1} {x_center} {y_center} {w} {h}"
        lines.append(line)  # Append line to the list
    
    yolo_lines = "\n".join(lines)  # Join lines with newline characters
    
    return yolo_lines

def make_yolo_annotations(input_dir, output_dir):
    """
    this function will create all the yolo annotations for the dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = list_files(input_dir, "xml")
    
    for xml in metadata:
        image_name = os.path.basename(xml).split(".")[0]
        image_size, boxes = process_xml(xml)
        yolo_boxes = []
        for box in boxes:
            yolo_box = transf_any_box(box, input_type="xyxy", output_type="yolo")
            yolo_box = relative(image_size[0], image_size[1], yolo_box)
            yolo_box = fix_bounds_relative(yolo_box, "yolo")
            yolo_boxes.append(yolo_box)
            
        yolo_line = process_yolo_box(yolo_boxes)
        
        output_path = os.path.join(output_dir, f"{image_name}.txt")
        with open(output_path, "w") as f:
            f.write(yolo_line)


def split_yolo_dataset(images_input_dir, annot_input_dir, output_dir, train_size=0.8, val_size=0.1):
    
    """
    this function will split the dataset into train, val and test and create the structure required by YOLO
    """
    os.makedirs(output_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    
    images = sorted(list_files(images_input_dir, "jpg"))
    annotations = sorted(list_files(annot_input_dir, "txt"))
    
    combined = list(zip(images, annotations))
    shuffle(combined)
    images, annotations = zip(*combined)
    
    train_examples = int(len(images) * train_size)
    val_examples = int(len(images) * val_size)


    split_sizes = {
            "train": [0, train_examples],
            "val": [train_examples, train_examples + val_examples],
            "test": [train_examples + val_examples, len(images)]
        }
        
    for split in splits:
        images_split_dir = os.path.join(output_dir, "images", split)
        annotations_split_dir = os.path.join(output_dir, "labels", split)
        os.makedirs(images_split_dir, exist_ok=True)
        os.makedirs(annotations_split_dir, exist_ok=True)
        
        
        start_idx, end_idx = split_sizes[split]
        
        for image, annotation in zip(images[start_idx:end_idx], annotations[start_idx:end_idx]):
            shutil.copy(image, images_split_dir)
            shutil.copy(annotation, annotations_split_dir)
        
 
def read_yolo_annotation(file):
    
    # open the file
    with open(file, "r") as f:
        lines = f.readlines()
    
    # save the lines in a list, witoout the \n and the first element (class)
    boxes = []
    for line in lines:
        line = line.strip().split(" ")
        boxes.append(line[1:])
    
    # convert the boxes to float
    boxes = [[float(x) for x in box] for box in boxes]
    return boxes
             

def make_dataframe(images_input_dir, annotations_input_dir, xml_dir ,output_path, format="csv", remove_base_dir="/home/ec2-user/dev/data/logodet3k/"):
    
    """
    this function will create a dataframe with the following columns:
    path, width, height, boxes, yolo_boxes
    """

    # make the directories for the ouput path if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    images = sorted(list_files(images_input_dir, "jpg")) # this will be one of the splits  (train, val, test)
    annotations = sorted(list_files(annotations_input_dir, "txt")) # this will be one of the splits  (train, val, test)
    # for all the images in this split we need to find the corresponding xml file
    data = []
    for image, annot in zip(images, annotations):
        # check if the image name is the same as the xml name without the extension
        image_name = os.path.basename(image).split(".")[0]
        annot_name = os.path.basename(annot).split(".")[0]
        
        # get the corresponding xml file
        xml = os.path.join(xml_dir, f"{annot_name}.xml")
        
        assert image_name == annot_name, f"Image {image_name} does not match xml {annot_name}"
        
        path = image
        image_size, xyxy_boxes = process_xml(xml)
        yolo_boxes = read_yolo_annotation(annot)
        xyxy_boxes_relative = []
        for box in xyxy_boxes:
            rel_box = relative(image_size[0], image_size[1], box)
            rel_box = fix_bounds_relative(rel_box, "xyxy")
            xyxy_boxes_relative.append(rel_box)
        
        
        path = path.replace(remove_base_dir, "")
            
        data.append([image_name, path, image_size[0], image_size[1], xyxy_boxes, yolo_boxes, xyxy_boxes_relative])
        
    
    df = pd.DataFrame(data, columns=["asset", "path", "width", "height", "xyxy_boxes", "yolo_boxes", "xyxy_boxes_relative"])
    
    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "parquet":
        df.to_parquet(output_path, index=False)
      
    return df


# if __name__ == "__main__":
    # unzip_file("/home/ec2-user/dev/data/logodet3k/LogoDet-3K.zip", "/home/ec2-user/dev/data/logodet3k/unzip")
    # input_dir = "/home/ec2-user/dev/data/logodet3k/unzip"
    # mv_to_dir(input_dir, "/home/ec2-user/dev/data/logodet3k/images", "jpg", True)
    # mv_to_dir(input_dir, "/home/ec2-user/dev/data/logodet3k/labels", "xml", True)

    # make_yolo_annotations("/home/ec2-user/dev/data/logodet3k/labels", "/home/ec2-user/dev/data/logodet3k/yolo_labels") 
        
    # split_yolo_dataset("/home/ec2-user/dev/data/logodet3k/images", "/home/ec2-user/dev/data/logodet3k/yolo_labels", "/home/ec2-user/dev/data/logodet3k/yolo_dataset", train_size=0.8, val_size=0.1)

    # make_dataframe("/home/ec2-user/dev/data/logodet3k/yolo_dataset/images/train", "/home/ec2-user/dev/data/logodet3k/yolo_dataset/labels/train", "/home/ec2-user/dev/data/logodet3k/labels", "/home/ec2-user/dev/data/logodet3k/yolo_dataset/train.csv", format="csv")
    # make_dataframe("/home/ec2-user/dev/data/logodet3k/yolo_dataset/images/val", "/home/ec2-user/dev/data/logodet3k/yolo_dataset/labels/val", "/home/ec2-user/dev/data/logodet3k/labels", "/home/ec2-user/dev/data/logodet3k/yolo_dataset/val.csv", format="csv")
    # make_dataframe("/home/ec2-user/dev/data/logodet3k/yolo_dataset/images/test", "/home/ec2-user/dev/data/logodet3k/yolo_dataset/labels/test", "/home/ec2-user/dev/data/logodet3k/labels", "/home/ec2-user/dev/data/logodet3k/yolo_dataset/test.csv", format="csv")

    # compress_dir("/home/ec2-user/dev/data/logodet3k/yolo_dataset")
