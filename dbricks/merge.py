import sys
import os
from utils.boxes import transf_any_box, fix_bounds_relative
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, lit, size, array_union
import pyspark.sql.functions as F
from pyspark.sql.types import LongType, ArrayType, StructType, StructField, DoubleType, StringType
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ast

def main():
    spark = SparkSession.builder.appName("DataProcessing").getOrCreate()
    
    output_schema = ArrayType(StructType([
        StructField("category", LongType(), True),
        StructField("x", DoubleType(), True),
        StructField("y", DoubleType(), True),
        StructField("width", DoubleType(), True),
        StructField("height", DoubleType(), True)
    ]))
    
    output_schema_ = StringType()
    
    def transform_boxes(boxes, input_type, output_type, class_id=0):
        if len(boxes) == 0:
            return []
        transformed_boxes = []
        if isinstance(boxes, str):
            boxes = ast.literal_eval(boxes)
        for box in boxes:
            transf_box = transf_any_box(box, input_type, output_type)
            transf_box = fix_bounds_relative(transf_box, output_type)
            if output_type in ["yolo", "xywh"]:
                x, y, w, h = transf_box
                box_dict = {"category": class_id, "x": x, "y": y, "width": w, "height": h}
            elif output_type == "xyxy":
                x1, y1, x2, y2 = transf_box
                box_dict = {"category": class_id, "x": x1, "y": y1, "x2": x2, "y2": y2}
            else:
                raise ValueError("Box type not supported")
            transformed_boxes.append(box_dict)
        return transformed_boxes
    
    def yolo_annotations(boxes_dict):
        if len(boxes_dict) == 0:
            return ""
        content = ""
        for box in boxes_dict:
            category = box.category
            x, y, w, h = box.x, box.y, box.width, box.height
            line = f"{category} {x} {y} {w} {h}\n"
            content += line
        return content
    
    def make_patch_format(box, width, height):
        x_c, y_c, w, h = box
        x1 = x_c - w/2
        y1 = y_c - h/2
        x1 = x1 * width
        y1 = y1 * height
        w = w * width
        h = h * height
        return x1, y1, w, h
    
    def plot_boxes(uri, boxes):
        img = Image.open(uri)
        img = img.convert("RGB")
        width, height = img.size
        fig, ax = plt.subplots(1)
        ax.imshow(img, cmap='gray')
        if boxes:
            for row in boxes:
                box = row.x, row.y, row.width, row.height
                x1, y1, w, h = make_patch_format(box, width, height)
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="r", facecolor='none')
                ax.add_patch(rect)
        plt.show()
    
    def transform_yolo_boxes(df):
        return df.withColumn("yolo_boxes_dict", transform_boxes_udf("yolo_boxes", lit("yolo"), lit("yolo"), lit(1)))

    def merge_boxes(df):
        return df.withColumn("merged_yolo_boxes", array_union("yolo_boxes_dict", "yolo_face_boxes_dict"))

    def read_parquet(path, spark):
        df = spark.read.parquet(path)
        return df

    def read_csv(path, spark):
        df = spark.read.csv(path, header=True)
        return df

    def save_dataframe_as_parquet(df, save_path):
        df.write.mode("overwrite").parquet(save_path)

    def save_dataframe_as_csv(df, save_path):
        df = df.select("asset", "uri", "width", "height", "yolo_annotations")
        df.repartition(1).write.mode("overwrite").csv(save_path, header=True)

    def add_yolo_annotations(df):
        return df.withColumn("yolo_annotations", yolo_annotations_udf("merged_yolo_boxes"))
    
    # udfs
    transform_boxes_udf = udf(transform_boxes, output_schema)
    yolo_annotations_udf = udf(yolo_annotations, output_schema_)
    
    
    path_faces_parquet = "/mnt/innovation/pdacosta/data/logodet_3k/ias/faces/"
    path_logodet3k_csv = "/mnt/innovation/pdacosta/data/logodet_3k/yolo_dataset"
    train_path = os.path.join(path_logodet3k_csv, "train.csv")
    val_path = os.path.join(path_logodet3k_csv, "val.csv")
    test_path = os.path.join(path_logodet3k_csv, "test.csv")
    
    faces_df = read_parquet(path_faces_parquet, spark)
    faces_df = faces_df.withColumnRenamed("boxes", "face_boxes")
    
    faces_df = faces_df.withColumn("yolo_face_boxes_dict", transform_boxes_udf("face_boxes", F.lit("xyxy"), F.lit("yolo"), F.lit(0)))
    
    train_df = read_csv(train_path, spark)
    val_df = read_csv(val_path, spark)
    test_df = read_csv(test_path, spark)
    
    train_df = transform_yolo_boxes(train_df)
    val_df = transform_yolo_boxes(val_df)
    test_df = transform_yolo_boxes(test_df)
    
    train_df = train_df.join(faces_df, on="asset", how="left")
    val_df = val_df.join(faces_df, on="asset", how="left")
    test_df = test_df.join(faces_df, on="asset", how="left")
    
    train_df = merge_boxes(train_df)
    val_df = merge_boxes(val_df)
    test_df = merge_boxes(test_df)
    
    
    
    train_df = add_yolo_annotations(train_df)
    val_df = add_yolo_annotations(val_df)
    test_df = add_yolo_annotations(test_df)
    
    # selected_row = train_df.select("merged_yolo_boxes", "uri").where(size("yolo_face_boxes_dict") > 0).first()
    # uri = selected_row.uri
    # boxes = selected_row.merged_yolo_boxes
    
    # uri = uri.replace("s3://mls.us-east-1.innovation/", "/dbfs/mnt/innovation/")
    # plot_boxes(uri, boxes)
    
    base_path = "/mnt/innovation/pdacosta/data/logodet_3k/annotations"
    
    save_dataframe_as_parquet(train_df, os.path.join(base_path, "parquet", "train"))
    save_dataframe_as_parquet(val_df, os.path.join(base_path, "parquet", "val"))
    save_dataframe_as_parquet(test_df, os.path.join(base_path, "parquet", "test"))
    
    save_dataframe_as_csv(train_df, os.path.join(base_path, "csv", "train"))
    save_dataframe_as_csv(val_df, os.path.join(base_path, "csv", "val"))
    save_dataframe_as_csv(test_df, os.path.join(base_path, "csv", "test"))

if __name__ == "__main__":
    main()
