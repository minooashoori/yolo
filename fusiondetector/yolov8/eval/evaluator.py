import os
import sys
from fusiondetector import PROJECT_DIR, OUTPUTS_DIR
sys.path.append(os.path.join(PROJECT_DIR, "ultralytics"))
# sys.path.append("/home/ec2-user/dev/ctx-logoface-detector/ultralytics")
import argparse
import glob


from PIL import Image
from tqdm import tqdm

from ultralytics import YOLO
from fusiondetector.utils.boxes import transf_any_box
from fusiondetector.metrics import pascalvoc


def map_class(class_name):
    class_mapping = {"face": 0, "logo": 1}
    return class_mapping.get(class_name, None)

def load_image_files(folder):
    image_extensions = (".jpg", ".png")
    return [file for file in glob.glob(os.path.join(folder, "*")) if file.endswith(image_extensions)]

def process_image(show, res, c_):

    content = ""

    if show:
        im_array = res.plot()
        im = Image.fromarray(im_array[..., ::-1])
        
    cl = res.boxes.cls.to("cpu").tolist()
    conf = res.boxes.conf.to("cpu").tolist()
    boxes_xywh = res.boxes.xywh.to("cpu").tolist()

    for c, b, xywh in zip(cl, conf, boxes_xywh):
        if c == c_:
            xywh = transf_any_box(xywh, "yolo", "xywh")
            xywh = [int(round(x)) for x in xywh]
            content += f"{str(int(c))} {round(b, 2)} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n"

    return im, content

def inference(args):
    c_ = map_class(args.c)

    if c_ is None:
        raise ValueError(f"Invalid class {args.c}")

    model = YOLO(args.model)
    results = model(args.imgs, stream=True, conf=0.12, verbose=False, half=True, device="cuda:0")
    show = args.show

    os.makedirs(args.det, exist_ok=True)
    files = load_image_files(args.imgs)
    n_files = len(files)

    print(f"Running inference on {args.det} containing {n_files} image files for class {c_}...")


    pbar = tqdm(total=n_files, unit="images")
    count = 0
    img_list = []
    for res in results:
        im, content = process_image(show, res, c_)
        img_list.append(im)

        filename = os.path.splitext(os.path.basename(res.path))[0]

        with open(os.path.join(args.det, f"{filename}.txt"), "w") as f:
            f.write(content)

        pbar.update(1)
        count += 1
        if count == 500:
            break

    if img_list:
        # do this for 5 batches of 25 images
        batch_size = 25
        num_batches = 5
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_images = img_list[start_idx:end_idx]

            if not batch_images:
                continue  # Skip empty batches

            grid_size = 2048
            cell_size = grid_size // 5
            grid = Image.new("RGB", (grid_size, grid_size), "white")

            for row in range(5):
                for col in range(5):
                    img_idx = row * 5 + col
                    if img_idx < len(img_list):
                        img = batch_images[img_idx]
                        img = img.resize((cell_size, cell_size))
                        grid.paste(img, (col * cell_size, row * cell_size))

            pred_path = os.path.join(OUTPUTS_DIR, f"{args.c}s_{batch_idx+1}.jpg")
            grid.save(pred_path)
            print(f"Saved sample predictions to {pred_path}")

    print(f"Saved {count} inference files to {args.det}")

def main(args):
    inference(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/ec2-user/dev/yolo/runs/detect/train14/weights/best.pt")
    # parser.add_argument("--imgs", type=str, default="/home/ec2-user/dev/data/logo05/yolo/images/test")
    parser.add_argument("--imgs", type=str, default="/home/ec2-user/dev/data/widerface/unzip/val")
    # parser.add_argument("--gt", type=str, default="/home/ec2-user/dev/data/logo05/annotations/gts_preds/gts")
    parser.add_argument("--gt", type=str, default="/home/ec2-user/dev/data/widerface/gts_preds/gts")
    # parser.add_argument("--det", type=str, default="/home/ec2-user/dev/data/logo05/annotations/gts_preds/preds_yolov8s_t14")
    parser.add_argument("--det", type=str, default="/home/ec2-user/dev/data/widerface/gts_preds/preds_yolov8s_t14")
    parser.add_argument("--c", type=str, default="face")
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    main(args)

    pascalvoc_parser = pascalvoc.parser
    pascalvoc_args_input = ["-gt", args.gt, "-det", args.det, "-conf", str(args.conf), "-t", str(args.iou)]

    pascalvoc_args = pascalvoc_parser.parse_args(pascalvoc_args_input)

    pascalvoc.main(pascalvoc_args)

