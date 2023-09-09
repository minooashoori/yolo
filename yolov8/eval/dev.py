import os
from ultralytics import YOLO, settings
import glob
from PIL import Image
from utils.boxes import transf_any_box

IMG_FOLDER = "/home/ec2-user/dev/data/logo05/yolo/images/test"
OUTPUT_FOLDER = "/home/ec2-user/dev/data/logo05/annotations/gts_preds/preds_yolo"


model = YOLO('/home/ec2-user/dev/yolo/runs/detect/train39/weights/best.pt') # yolom trained on all the data+logodet3k
batch_size = 1
results = model(IMG_FOLDER, stream=True, conf=0.20)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for res in results:
    # print(res.boxes)
    # im_array = res.plot()  # plot a BGR numpy array of predictions
    # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    # im.save('results.jpg')  # save image
    
    cls = res.boxes.cls.to("cpu").tolist()
    conf = res.boxes.conf.to("cpu").tolist()
    boxes_xywh = res.boxes.xywh.to("cpu").tolist()
    content = ""
    for c, b, xywh in zip(cls, conf, boxes_xywh):
        if c == 1:

            xywh = transf_any_box(xywh, "yolo", "xywh")
            xywh = [int(round(x)) for x in xywh]
            content += f"{str(int(c))} {round(b,2)} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n"

    # get name without extension from the path
    filename = os.path.splitext(os.path.basename(res.path))[0]

    with open(os.path.join(OUTPUT_FOLDER, filename+".txt"), "w") as f:
        f.write(content)
        

