import os
from dbutils.paths import innovation_path
from dbutils.annotations import read_boxes_parquet
from pyspark.sql.types import LongType, ArrayType, StructType, StructField, DoubleType, StringType
from pyspark.sql.functions import udf, lit, size, array_union


WIDERFACE_PATH = os.path.join(innovation_path("mnt"), "pdacosta", "data", "wider_face")
INPUT_PATH_GTS = os.path.join(WIDERFACE_PATH, "dataset/val.csv")
INPUT_PATH_SCORES = os.path.join(WIDERFACE_PATH, "preds_faces/xywh/")
OUTPUT_PATH_PARQUET = os.path.join(WIDERFACE_PATH, "gts_preds/xywh")


def rel_to_abs(box, width, height):
    x, y, w, h = box
    x = x * width
    y = y * height
    w = w * width
    h = h * height
    return int(x), int(y), int(w), int(h)

def format_gt_boxes(boxes_dict, width, height, abs):
    if len(boxes_dict) == 0:
        return ""
    content = ""
    for box in boxes_dict:
        category = box.category
        x, y, w, h = box.x, box.y, box.width, box.height
        if abs:
            x, y, w, h = rel_to_abs([x, y, w, h], width, height)
        line = f"{category} {x} {y} {w} {h}\n"
        content += line
    return content

def format_pred_boxes(boxes_list, width, height, abs, category="0"):
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
        if abs:
            box = rel_to_abs(box, width, height)
        box = [round(coord, 4) for coord in box]
        score = round(box_dict["score"], 2)
        line = f"{category} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n"
        content += line
    return content


if __name__ == "__main__":
    
    # read gt
    gt_df = spark.read.parquet(INPUT_PATH_TRAIN).select("asset", "width", "height", "xywh")
    gt_df.show()

    # read preds
    # scores_train_df = spark.read.parquet(INPUT_PATH_SCORES_TRAIN).withColumnRenamed("boxes", "xywh_pred_boxes").withColumnRenamed("asset", "asset_id")
    # scores_test_df = spark.read.parquet(INPUT_PATH_SCORES_TEST).withColumnRenamed("boxes", "xywh_pred_boxes").withColumnRenamed("asset", "asset_id")
    
    # # join gt and preds
    # train_df = train_df.join(scores_train_df, on="asset_id", how="left")
    # test_df = test_df.join(scores_test_df, on="asset_id", how="left")
    
    # # check if we  have preds for all assets
    # train_df = train_df.withColumn("preds", size("xywh_pred_boxes"))
    # test_df = test_df.withColumn("preds", size("xywh_pred_boxes"))

    # # count assets with no preds
    # print("nopreds train:", train_df.filter("preds == 0").count())
    # print("nopreds test:", test_df.filter("preds == 0").count())
    
    
    # # # make udfs
    # format_gt_boxes_udf = udf(format_gt_boxes, StringType())
    # format_pred_boxes_udf = udf(format_pred_boxes, StringType())
    
    # # # apply udfs - train
    # train_df = train_df.withColumn("gt_boxes", format_gt_boxes_udf("xywh_boxes", "width", "height", lit(False)))
    # train_df = train_df.withColumn("pred_boxes", format_pred_boxes_udf("xywh_pred_boxes", "width", "height", lit(False)))

    # train_df = train_df.withColumn("gt_boxes_abs", format_gt_boxes_udf("xywh_boxes", "width", "height", lit(True)))
    # train_df = train_df.withColumn("pred_boxes_abs", format_pred_boxes_udf("xywh_pred_boxes", "width", "height", lit(True)))
    # # # # apply udfs - test
    # test_df = test_df.withColumn("gt_boxes", format_gt_boxes_udf("xywh_boxes", "width", "height", lit(False)))
    # test_df = test_df.withColumn("pred_boxes", format_pred_boxes_udf("xywh_pred_boxes", "width", "height", lit(False)))

    # test_df = test_df.withColumn("gt_boxes_abs", format_gt_boxes_udf("xywh_boxes", "width", "height", lit(True)))
    # test_df = test_df.withColumn("pred_boxes_abs", format_pred_boxes_udf("xywh_pred_boxes", "width", "height", lit(True)))
    

    # # # # select columns
    # train_df = train_df.select("asset_id", "width", "height", "uri", "gt_boxes", "pred_boxes", "gt_boxes_abs", "pred_boxes_abs")
    # test_df = test_df.select("asset_id", "width", "height", "uri", "gt_boxes", "pred_boxes", "gt_boxes_abs", "pred_boxes_abs")
    
    # # test_df.show()

    # # # # save as parquet
    # train_df.write.mode("overwrite").parquet(OUTPUT_PATH_TRAIN_PARQUET)
    # test_df.write.mode("overwrite").parquet(OUTPUT_PATH_TEST_PARQUET)

    # # # save just one file for each dataset
    # train_df.coalesce(1).write.mode("overwrite").csv(OUTPUT_PATH_TRAIN, header=True)
    # test_df.coalesce(1).write.mode("overwrite").csv(OUTPUT_PATH_TEST, header=True)