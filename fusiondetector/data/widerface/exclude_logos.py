import sys
import os
from fusiondetector import PROJECT_DIR, OUTPUTS_DIR
from fusiondetector.yolov8.eval.evaluator import map_class, load_image_files, process_image
sys.path.append(os.path.join(PROJECT_DIR, "ultralytics"))
from ultralytics import YOLO
import pandas as pd
import ast
import shutil


def process_image_metadata(res, metadata, c_, output_labels, output_images):

    content = ""
    cl = res.boxes.cls.to("cpu").tolist()
    conf = res.boxes.conf.to("cpu").tolist()
    boxes_yolo = res.boxes.xywh.to("cpu").tolist()
    path = res.path


    include_im = True
    for c, b, yolo_box in zip(cl, conf, boxes_yolo):
        if c == c_:

            include_im = False
            break

    if include_im:
        # get the filename
        filename = os.path.basename(path)
        # maatch the filename with the metadata column "path" format: train/0_Parade_marchingband_1_849.jpg
        metadata_row = metadata[metadata["path"].str.contains(filename)]
        # get the column "yolo_boxes"
        yolo_boxes = metadata_row["yolo_boxes"]
        yolo_boxes = ast.literal_eval(yolo_boxes.values[0])
        for box in yolo_boxes:
            content += f"{str(0)} {box[0]} {box[1]} {box[2]} {box[3]}\n"

        # write the content to the file with the same file name as the image but with .txt extension
        with open(os.path.join(output_labels, filename.replace(".jpg", ".txt")), "w") as f:
            f.write(content)
        # copy the image over
        shutil.copy(path, output_images)

        # we need to get the metadata for the current image





def inference(w: str, conf:float = 0.2, device="cuda:0",
    ims: str = "/home/ec2-user/dev/data/widerface/unzip/train",
              metadata: str = "/home/ec2-user/dev/data/widerface/metadata/train.csv",
            output_labels = "/home/ec2-user/dev/data/widerface/yolo/labels/train",
            output_images = "/home/ec2-user/dev/data/widerface/yolo/images/train"):
    
    """This function will run the detector for all images of WIDERFace and will remove any images that contain logos.

    Args:
        ims (str): path to the images folder
    """
    c_ = 1.0 # we will exclude logos
    model = YOLO(w)
    results = model(ims, stream=True, conf=conf, verbose=False, half=True, device=device)
    # ims = load_image_files(ims)
    metadata = pd.read_csv(metadata)

    # delete the output folders if they exist
    if os.path.exists(output_labels):
        shutil.rmtree(output_labels)
    if os.path.exists(output_images):
        shutil.rmtree(output_images)

    os.makedirs(output_labels, exist_ok=True)
    os.makedirs(output_images, exist_ok=True)

    for res in results:
        process_image_metadata(res, metadata, c_, output_labels, output_images)


# inference(w="/home/ec2-user/dev/yolo/runs/detect/train5/weights/epoch12.pt")


img = "/home/ec2-user/dev/data/widerface/unzip/train/37_Soccer_soccer_ball_37_620.jpg"
model = YOLO("/home/ec2-user/dev/yolo/runs/detect/train82/weights/best.pt")
results = model(img, conf=0.01, verbose=True, half=False, device="cuda:0")
# orig_img = results.orig_img
# import matplotlib.pyplot as plt
# #save the image
# plt.imshow(orig_img)
# plt.savefig("/home/ec2-user/dev/data/widerface/yolo/images/train/37_Soccer_Soccer_37_792.jpg")
print(results)