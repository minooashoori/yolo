import os
from utils.io import download, unzip_file, list_files, mv_to_dir, list_files
from databricks.preprocess.boxes import transf_any_box, relative, fix_bounds_relative
import PIL.Image as Image
import pandas as pd

def retrieve_boxes(metadata_path, remove_dir=True):


    """
    Open and process the metadata file. It is a txt file with the following structure:
    9--Press_Conference/9_Press_Conference_Press_Conference_9_129.jpg
    2
    336 242 152 202 0 0 0 0 0 0
    712 278 126 152 0 0 0 0 0 0
    9--Press_Conference/9_Press_Conference_Press_Conference_9_183.jpg
    3
    218 190 112 160 0 0 0 0 0 0
    302 224 74 140 0 0 0 0 0 0
    560 136 170 204 1 0 0 0 0 0
    """
    # open txt file
    with open(metadata_path, "r") as f:
        lines = f.readlines()

    boxes = {}
    # process the lines
    for line in lines:
        if line.endswith(".jpg\n"):
            # This is a file path
            file_path = line.strip()
            # remove the first folder name (e.g. 9--Press_Conference)
            if remove_dir:
                file_path = os.path.join(*file_path.split("/")[1:])
            boxes[file_path] = []

        elif line.strip().isdigit():
            pass
        else:
            # x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose%
            x, y, w, h, *_ = [int(x) for x in line.strip().split(" ")]
            # we need to save the boxes
            box = [int(x), int(y), int(w), int(h)]
            boxes[file_path].append(box)


    return boxes


def save_metadata(boxes, imgs_dir, output_path, remove_base_dir, save_as="csv", input_type="xywh", output_type="yolo", normalize=True ):

    # ensure the format of the file in output_path matehces the save_as parameter
    if not output_path.endswith(f".{save_as}"):
        output_path = f"{output_path}.{save_as}"

    # Create a list to store metadata for each image
    metadata = []

    for img_path, boxes in boxes.items():
        # Open the image to get the width and height
        asset = img_path
        img_path = os.path.join(imgs_dir, img_path)
        img = Image.open(img_path)
        width, height = img.size

        # Process the boxes
        new_boxes = []
        old_boxes = []
        for box in boxes:
            if box[0] == 0 and box[1] == 0 and box[2] == 0 and box[3] == 0:
                continue

            new_box = transf_any_box(box, input_type, output_type)
            if normalize:
                new_box = relative(width, height, new_box)
                new_box = fix_bounds_relative(new_box, output_type)
            new_boxes.append(new_box)
            old_boxes.append(box)

        if remove_base_dir:
            img_path = img_path.replace(remove_base_dir, "")

        # Append metadata for the current image to the list
        metadata.append({
            "asset": asset,
            "path": img_path,
            "width": width,
            "height": height,
            "xywh": old_boxes,
            f"{output_type}_boxes": new_boxes
        })

    # Create a DataFrame from the metadata list
    df = pd.DataFrame(metadata)

    save_dir = os.path.dirname(output_path)
    # create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    if save_as == "csv":
        df.to_csv(output_path, index=False)
    elif save_as == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        print("Unsupported save format.")

    print(f"Processed metadata saved to {output_path}")








if __name__ == "__main__":
    os.environ["AWS_PROFILE"] = "saml"
    # Download the dataset
    base_s3_path = "s3://mls.us-east-1.innovation/pdacosta/data/wider_face/zip"
    s3_path_train = os.path.join(base_s3_path, "WIDER_train.zip")
    s3_path_val = os.path.join(base_s3_path, "WIDER_val.zip")
    s3_path_annotations = os.path.join(base_s3_path, "wider_face_split.zip")
    local_path = "/home/ec2-user/dev/data/widerface/zip"
    s3_paths = [s3_path_train, s3_path_val, s3_path_annotations]
    for s3_path in s3_paths:
        download(s3_path, local_path)
    local_paths = [os.path.join(local_path, os.path.basename(s3_path)) for s3_path in s3_paths]

    # for local_path in local_paths:
    #     unzip_file(local_path)

    # Move the files to the right place
    # mv_to_dir( "/home/ec2-user/dev/data/widerface/zip/WIDER_train", "/home/ec2-user/dev/data/widerface/unzip/train", "jpg", rename=False, keep_dir_structure=False)
    # mv_to_dir( "/home/ec2-user/dev/data/widerface/zip/WIDER_val", "/home/ec2-user/dev/data/widerface/unzip/val", "jpg", rename=False, keep_dir_structure=False)

    # # Process the metadata
    boxes_train = retrieve_boxes("/home/ec2-user/dev/data/widerface/zip/wider_face_split/wider_face_train_bbx_gt.txt")
    boxes_val = retrieve_boxes("/home/ec2-user/dev/data/widerface/zip/wider_face_split/wider_face_val_bbx_gt.txt")

    save_metadata(boxes_train,
                  "/home/ec2-user/dev/data/widerface/unzip/train",
                  "/home/ec2-user/dev/data/widerface/metadata/train.csv",
                  "/home/ec2-user/dev/data/widerface/unzip/",
                  save_as="csv", input_type="xywh", output_type="yolo", normalize=True)

    save_metadata(boxes_val,
                "/home/ec2-user/dev/data/widerface/unzip/val",
                "/home/ec2-user/dev/data/widerface/metadata/val.csv",
                "/home/ec2-user/dev/data/widerface/unzip/",
                save_as="csv", input_type="xywh", output_type="yolo", normalize=True)
    # save_metadata(boxes_val, "/home/ec2-user/dev/data/widerface/unzip/val", "/home/ec2-user/dev/data/widerface/metadata/val", save_as="csv", input_type="xywh", output_type="yolo", normalize=True)
    # images = sorted(list_files("/home/ec2-user/dev/data/widerface/unzip/train", "jpg")) # this will be one of the splits  (train, val, test)
    # print(images[:10])