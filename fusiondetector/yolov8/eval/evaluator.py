import os
import sys

sys.path.append("/home/ec2-user/dev/ctx-logoface-detector/ultralytics")
import argparse
import glob

from PIL import Image
from tqdm import tqdm

from ultralytics import YOLO
from utils.boxes import transf_any_box


def map_class(class_name):
    class_mapping = {"face": 0, "logo": 1}
    return class_mapping.get(class_name, None)

def load_image_files(folder):
    image_extensions = (".jpg", ".png")
    return [file for file in glob.glob(os.path.join(folder, "*")) if file.endswith(image_extensions)]

def process_image(args, res, c_):
    pil_images = []
    content = ""

    if args.show:
        im_array = res.plot()
        im = Image.fromarray(im_array[..., ::-1])
        pil_images.append(im)

    cl = res.boxes.cls.to("cpu").tolist()
    conf = res.boxes.conf.to("cpu").tolist()
    boxes_xywh = res.boxes.xywh.to("cpu").tolist()

    for c, b, xywh in zip(cl, conf, boxes_xywh):
        if c == c_:
            xywh = transf_any_box(xywh, "yolo", "xywh")
            xywh = [int(round(x)) for x in xywh]
            content += f"{str(int(c))} {round(b, 2)} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n"

    return pil_images, content

def inference(args):
    c_ = map_class(args.c)

    if c_ is None:
        raise ValueError(f"Invalid class {args.c}")

    model = YOLO(args.model_path)
    results = model(args.img_folder, stream=True, conf=args.conf, verbose=False, half=True, device="cuda")

    os.makedirs(args.output_folder, exist_ok=True)
    files = load_image_files(args.img_folder)
    n_files = len(files)

    print(f"Running inference on {args.img_folder} containing {n_files} image files for class {c_}...")

    pil_images = []
    count = 0
    pbar = tqdm(total=n_files, unit="images")

    for res in results:
        img_list, content = process_image(args, res, c_)
        pil_images.extend(img_list)

        filename = os.path.splitext(os.path.basename(res.path))[0]

        with open(os.path.join(args.output_folder, f"{filename}.txt"), "w") as f:
            f.write(content)

        count += 1
        pbar.update(1)

    if pil_images:
        grid_size = 2048
        cell_size = grid_size // 5
        grid = Image.new("RGB", (grid_size, grid_size), "white")

        for row in range(5):
            for col in range(5):
                img_idx = row * 5 + col
                if img_idx < len(pil_images):
                    img = pil_images[img_idx]
                    img = img.resize((cell_size, cell_size))
                    grid.paste(img, (col * cell_size, row * cell_size))
        pred_path = os.path.join(os.getcwd(), "pred.jpg")
        grid.save(pred_path)
        print(f"Saved sample predictions to {pred_path}")

    print(f"Saved {count} inference files to {args.output_folder}")

def main(args):
    inference(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/ec2-user/dev/yolo/saved/yolov8m_t0_epoch4.pt")
    parser.add_argument("--img_folder", type=str, default="/home/ec2-user/dev/data/logo05/yolo/images/test")
    parser.add_argument("--output_folder", type=str, default="/home/ec2-user/dev/data/logo05/annotations/gts_preds/preds_yolov8m_t0")
    parser.add_argument("--c", type=str, default="logo")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--conf", type=float, default=0.0)
    args = parser.parse_args()

    main(args)