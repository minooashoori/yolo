import os
from ultralytics import YOLO
from PIL import Image
import glob
from utils.boxes import transf_any_box
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/home/ec2-user/dev/yolo/runs/detect/train/weights/epoch4.pt")
parser.add_argument("--model", type=str, default="train51")
parser.add_argument("--epoch", type=int, default=59)
parser.add_argument("--conf", type=float, default=0.01)
parser.add_argument("--img_folder", type=str, default="/home/ec2-user/dev/data/logo05/yolo/images/test")
# parser.add_argument("--img_folder", type=str, default="/home/ec2-user/dev/data/widerface/unzip/val")
parser.add_argument("--output_folder", type=str, default="/home/ec2-user/dev/data/logo05/annotations/gts_preds/preds_yolo_small")
# parser.add_argument("--output_folder", type=str, default="/home/ec2-user/dev/data/widerface/gts_preds/preds_yolo_medium")
parser.add_argument("--runs_folder", type=str, default="/home/ec2-user/dev/yolo/runs/detect")
parser.add_argument("--c", type=int, default=1)
parser.add_argument("--show", type=bool, default=True)
args = parser.parse_args()


def inference(args):
    if not args.model_path:
        model_path = f"{args.runs_folder}/{args.model}/weights/epoch{args.epoch}.pt"
    else:
        model_path = args.model_path
    model = YOLO(model_path)
    # results = model(args.img_folder, stream=True, conf=args.conf, verbose=False, half=True)
    results = model(args.img_folder, stream=True, conf=args.conf, verbose=False, half=False, device="cpu")
    os.makedirs(args.output_folder, exist_ok=True)
    # get the number of images: can be jpg or png
    files = glob.glob(os.path.join(args.img_folder, "*.jpg")) + glob.glob(os.path.join(args.img_folder, "*.png"))
    n_files = len(files)

    print(f"Running inference on {args.img_folder} containing {n_files} files for class {args.c}...")
    pil_images = []
    count = 0
    pbar = tqdm(total=n_files, unit="images")
    for res in results:
        if args.show:
            if len(pil_images) < 50:
                im_array = res.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                pil_images.append(im)

        cl = res.boxes.cls.to("cpu").tolist()
        conf = res.boxes.conf.to("cpu").tolist()
        boxes_xywh = res.boxes.xywh.to("cpu").tolist()
        content = ""
        for c, b, xywh in zip(cl, conf, boxes_xywh):
            if c == args.c:
                xywh = transf_any_box(xywh, "yolo", "xywh")
                xywh = [int(round(x)) for x in xywh]
                content += f"{str(int(c))} {round(b,2)} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n"

        # get name without extension from the path
        filename = os.path.splitext(os.path.basename(res.path))[0]

        with open(os.path.join(args.output_folder, filename+".txt"), "w") as f:
            f.write(content)
        count += 1
        pbar.update(1)


    if pil_images:
        # create a grid
        grid_size = 2048
        cell_size =  grid_size // 5
        grid = Image.new("RGB", (grid_size, grid_size), "white")
        for row in range(5):
            for col in range(5):
                img_idx = row * 5 + col
                if img_idx < len(pil_images):
                    img = pil_images[img_idx]
                    img = img.resize((cell_size, cell_size))
                    grid.paste(img, (col * cell_size, row * cell_size))
        pred_path = os.path.join(args.output_folder, "pred.jpg")
        grid.save(pred_path)
        print(f"Saved sample predictions to {pred_path}")
    print(f"Saved {count} inference files to {args.output_folder}")

def main():
    inference(args)

if __name__ == "__main__":
    main()
