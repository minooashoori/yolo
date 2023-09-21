import os
from dbutils.paths import innovation_path
from dbutils.annotations import read_boxes_parquet
from pyspark.sql.types import LongType, ArrayType, StructType, StructField, DoubleType, StringType
from pyspark.sql.functions import udf, lit, size, array_union, col
import shutil


WIDERFACE_PATH = os.path.join(innovation_path("mnt"), "pdacosta", "data", "wider_face")
INPUT_PATH_GTS = os.path.join(WIDERFACE_PATH, "dataset/val.csv")
INPUT_PATH_DETS = os.path.join(WIDERFACE_PATH, "preds_faces/xywh/")
OUTPUT_PATH_PARQUET = os.path.join(WIDERFACE_PATH, "gts_preds/parquet/xywh")
OUTPUT_PATH_CSV = os.path.join(WIDERFACE_PATH, "gts_preds/csv/xywh")


def rel_to_abs(box, width, height):

    x, y, x_or_w, y_or_h = box
    x = round(x * width)
    y = round(y * height)
    x_or_w = round(x_or_w * width)
    y_or_h = round(y_or_h * height)
    return int(x), int(y), int(x_or_w), int(y_or_h)

def format_gt_boxes(boxes, width, height, to_abs, category="0"):
    import ast
    # gt boxes is  a list of lists, but can be saved as a string
    # we need to format it back as a list of lists
    if boxes is None:
        return ""
    if isinstance(boxes, str):
        boxes = ast.literal_eval(boxes)
    if len(boxes) == 0:
        return ""
    content = ""
    for box in boxes:
        print(len(box))
        x, y, w, h = box
        if to_abs:
            x, y, w, h = rel_to_abs([x, y, w, h], width, height)
        line = f"{category} {x} {y} {w} {h}\n"
        content += line
    return content

def format_pred_boxes(boxes_list, width, height, to_abs, category="0"):
    # pred boxes is: [{[0.3644, 0.2538, 0.0259, 0.0538], 0.8626074194908142}, {[0.21, 0.6625, 0.0209, 0.0136], 0.17813876271247864}]
    # we need to format it as a string as: class_id score x y w h
    if boxes_list is None:
        return ""
    if len(boxes_list) == 0:
        return ""
    content = ""
    for box_dict in boxes_list:
        box = box_dict["box"]
        if box is None:
            continue
        if to_abs:
            box = rel_to_abs(box, width, height)
        score = round(box_dict["score"], 2)
        line = f"{category} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n"
        content += line
    return content


if __name__ == "__main__":

    # read gt
    gt_df = spark.read.csv(INPUT_PATH_GTS, header=True).select("asset", "width", "height", "xywh")
    # change format of width and height to int
    gt_df = gt_df.withColumn("width", gt_df["width"].cast(LongType()))
    gt_df = gt_df.withColumn("height", gt_df["height"].cast(LongType()))
    # gt_df.show(truncate=False)

    # read preds
    det_df = spark.read.parquet(INPUT_PATH_DETS).withColumnRenamed("boxes", "xywhn_det")
    # det_df.show()

    # join gt and preds
    df = gt_df.join(det_df, on="asset", how="inner")
    # df.show()

    # number of rows
    print(f"Number of rows: {df.count()}")
    # remove rows if we don't have gts
    df = df.filter("xywh is not null")
    # also remove if the preds are empty, which in this case it's a sting []
    df  = df.filter(col("xywh") != "[]")
    print(f"Number of rows after removing empty gts: {df.count()}")
    # make udfs
    format_gt_boxes_udf = udf(format_gt_boxes, StringType())
    format_pred_boxes_udf = udf(format_pred_boxes, StringType())

    # apply udfs
    df = df.withColumn("gt_xywh", format_gt_boxes_udf("xywh", "width", "height", lit(False)))
    df = df.withColumn("det_xywh", format_pred_boxes_udf("xywhn_det", "width", "height", lit(True)))

    # save as parquet
    df.write.mode("overwrite").parquet(OUTPUT_PATH_PARQUET)

    # save as csv
    df.select("asset", "uri", "width", "height", "gt_xywh", "det_xywh").write.mode("overwrite").csv(OUTPUT_PATH_CSV, header=True)
    df.printSchema()