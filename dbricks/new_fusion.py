from dbutils.boxes import transform_box, iou_yolo, transf_any_box, fix_bounds_relative, relative, yolo_annotations
from dbutils.paths import create_uris
from pyspark.sql.functions import udf
import pyspark.sql.functions as F
from pyspark.sql.types import LongType, ArrayType, StructType, StructField, DoubleType, StringType
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ast
from dbutils.paths import  innovation_path
import os

OUTPUT_SCHEMA_BOX = ArrayType(StructType([
    StructField("category", LongType(), True),
    StructField("x", DoubleType(), True),
    StructField("y", DoubleType(), True),
    StructField("width", DoubleType(), True),
    StructField("height", DoubleType(), True)
]))


def boxes_to_dict(boxes, width, height, input_format, output_format, class_id=0):
    """
    Convert bounding boxes to a list of dictionaries with specified output format.

    Args:
        boxes (str or list): List of bounding boxes in input format or a string.
        width (int): Image width.
        height (int): Image height.
        input_type (str): Input bounding box format ("xywh", "xyxy", "yolo").
        output_type (str): Desired output format ("yolo", "xywh", "xyxy").
        class_id (int): Category/class ID.

    Returns:
        list: List of dictionaries representing bounding boxes in the desired format.
    """
    if len(boxes) == 0:
        return []

    transformed_boxes = []

    if isinstance(boxes, str):
        boxes = ast.literal_eval(boxes)

    for box in boxes:
        # Transform to relative coordinates
        transf_box = relative(width, height, box)
        # Convert to the desired output format
        transf_box = transf_any_box(transf_box, input_format, output_format)
        transf_box = fix_bounds_relative(transf_box, output_format)

        if output_format in ["yolo", "xywh"]:
            x, y, w, h = transf_box
            box_dict = {"category": class_id, "x": x, "y": y, "width": w, "height": h}
        elif output_format == "xyxy":
            x1, y1, x2, y2 = transf_box
            box_dict = {"category": class_id, "x": x1, "y": y1, "x2": x2, "y2": y2}
        else:
            raise ValueError("Box type not supported")

        transformed_boxes.append(box_dict)

    return transformed_boxes


# def merge_boxes(retina_boxes, ias_boxes, iou_threshold=0.25):

#     if len(retina_boxes) == 0:
#         return ias_boxes
#     if len(ias_boxes) == 0:
#         return retina_boxes

#     # we check retina boxes against ias boxes, if there is an overlap we discard the retina box, else we keep it
#     include_boxes = []
#     for retina_box in retina_boxes:
#         x_r, y_r, w_r, h_r = retina_box["x"], retina_box["y"], retina_box["width"], retina_box["height"]
#         r_box = [x_r, y_r, w_r, h_r]
#         overlap = False
#         for ias_box in ias_boxes:
#             x_i, y_i, w_i, h_i = ias_box["x"], ias_box["y"], ias_box["width"], ias_box["height"]
#             i_box = [x_i, y_i, w_i, h_i]
#             # check if there is an overlap with at least one ias box
#             if iou_yolo(r_box, i_box) > iou_threshold:
#                 overlap = True
#                 break
#         if not overlap:
#             include_boxes.append(retina_box)

#     return include_boxes + ias_boxes

def merge_boxes(candidate_boxes, current_boxes, iou_threshold=0.25):
    """
    Merge and filter bounding boxes based on intersection over union (IOU) threshold.

    This function takes two lists of bounding boxes, 'candidate_boxes' and 'ias_boxes', and returns a new list of merged
    boxes. If a 'candidate_box' has an intersection over union (IOU) greater than or equal to the specified 'iou_threshold'
    with any 'keep_box', it is discarded; otherwise, it is included in the result.

    Args:
        candidate_boxes (list of dict): List of bounding boxes
        current_boxes (list of dict): List of bounding boxes
        iou_threshold (float, optional): The IOU threshold for overlap detection (default is 0.25).

    Returns:
        list of dict: A list of merged and filtered bounding boxes
    """
    if len(candidate_boxes) == 0:
        return current_boxes
    if len(current_boxes) == 0:
        return candidate_boxes

    include_boxes = []

    for candidate_box in candidate_boxes:
        x_r, y_r, w_r, h_r = candidate_box["x"], candidate_box["y"], candidate_box["width"], candidate_box["height"]
        r_box = [x_r, y_r, w_r, h_r]
        overlap = False

        for current_box in current_boxes:
            x_i, y_i, w_i, h_i = current_box["x"], current_box["y"], current_box["width"], current_box["height"]
            i_box = [x_i, y_i, w_i, h_i]

            # Check if there is an overlap with at least one ias box
            if iou_yolo(r_box, i_box) > iou_threshold:
                overlap = True
                break

        if not overlap:
            include_boxes.append(candidate_box)

    return include_boxes + current_boxes



def read_boxes_parquet(path, spark, alg_type:str):
    df = spark.read.parquet(path)
    df = df.drop("img_bytes").select("asset_id", "boxes")
    df = df.withColumnRenamed("boxes", f"{alg_type}_boxes")
    df = df.dropDuplicates(["asset_id"])
    return df


def undersample_faces(df, seed=42, prop=0.6, use_pseudo_labels=False):
    """
    Undersample the DataFrame to balance the number of faces and logos.

    This function takes a DataFrame 'df' containing information about assets, including whether they have faces and logos,
    and undersamples the assets with faces to match the proportion 'prop' of assets with logos. It ensures a balanced
    distribution of assets with both faces and logos, assets with only faces, and assets with only logos.

    Args:
        df (pyspark.sql.DataFrame): The input DataFrame with asset information.
        seed (int, optional): Seed for random undersampling (default is 42).
        prop (float, optional): The desired proportion of logos to maintain (default is 0.6).
        use_pseudo_labels (bool, optional): Set to True if using pseudo labels for logos (default is False).

    Returns:
        pyspark.sql.DataFrame: A DataFrame with undersampled assets.
    """
    if use_pseudo_labels:
        print("Undersampling faces based on pseudo logo labels...")
    else:
        print("Undersampling faces based on gt logo labels...")

    print(f"Undersampling with proportion {prop} of logos...")

    df = df.withColumn("has_face", F.when(F.size(F.col("face_boxes")) > 0, 1).otherwise(0))
    df = df.withColumn("has_logo", F.when(F.size(F.col("boxes")) > 0, 1).otherwise(0))

    if use_pseudo_labels:
        # check if "pseudo_boxes" column exists
        if "pseudo_boxes" not in df.columns:
            raise ValueError("pseudo_boxes column not found in DataFrame")
        df = df.withColumn("has_pseudologo", F.when(F.size(F.col("pseudo_boxes")) > 0, 1).otherwise(0))
        df_face = df.filter((F.col("has_face") == 1) & (F.col("has_pseudologo") == 0))
    else:
        df_face = df.filter((F.col("has_face") == 1) & (F.col("has_logo") == 0))

    n_assets_with_face = df_face.count()
    n_assets_with_logo = df.filter((F.col("has_face") == 0) & (F.col("has_logo") == 1)).count()

    print(f"Number of assets with ONLY faces: {n_assets_with_face}")
    print(f"Number of assets with ONLY logos: {n_assets_with_logo}")

    fraction_face_logo = min(prop * n_assets_with_logo / n_assets_with_face, 1.0)
    df_face = df_face.sample(False, fraction_face_logo, seed=seed)
    print(f"Number of assets with ONLY faces after undersampling: {df_face.count()}")

    df_neither = df.filter((F.col("has_face") == 0) & (F.col("has_logo") == 0))
    df_both = df.filter((F.col("has_face") == 1) & (F.col("has_logo") == 1))
    df_logo = df.filter((F.col("has_face") == 0) & (F.col("has_logo") == 1))

    df = df_both.union(df_neither).union(df_logo).union(df_face)

    print(f"Number of assets after undersampling faces: {df.count()}")

    return df



def undersample_backgrounds(df, prop=0.02):

    # split the rows that don't have a box
    # df_no_boxes = df.filter(F.size(F.col("boxes")) == 0)
    print(f"Undersampling backgrounds with proportion {prop} of logos/faces...")

    df_no_boxes = df.filter((F.col('has_face') == 0) & (F.col('has_logo') == 0))

    # split the rows that have a box
    df_boxes = df.filter(F.size(F.col("boxes")) > 0)

    print(f"Number of assets with boxes: {df_boxes.count()}")
    print(f"Number of assets without boxes (backgrounds): {df_no_boxes.count()}")
    # sample the assets without boxes to be max 2%*n_assets_with_boxes
    fraction = (prop*df_boxes.count())/df_no_boxes.count()
    fraction = min(fraction, 1.0)

    print(f"Keep {fraction*100:.2f}% of the assets without boxes")

    df_no_boxes = df_no_boxes.sample(False, fraction, seed=42)

    # stack the two dataframes again
    df = df_boxes.union(df_no_boxes)
    print("Final number of assets: {}".format(df.count()))
    df_no_boxes.unpersist()

    return df

def split_train_val_with_column(df, column: str):
    # Use a single filter operation to split the DataFrame
    train_df_logos = df.filter(F.col(column) == 1)
    val_df_logos = df.filter(F.col(column) == 0)

    # Count the rows directly
    n_train = train_df_logos.count()
    n_val = val_df_logos.count()

    print(f"Number of assets from logo gt train: {n_train}")
    print(f"Number of assets from logo gt val: {n_val}")

    total = n_train + n_val
    print(f"Total number of assets from logo gt: {total}")

    # Use a single filter operation for assets without logo gt
    df_no_logos = df.filter(F.col(column).isNull())

    print(f"Number of assets without logo gt: {df_no_logos.count()}")

    prop_train = n_train / total
    prop_val = n_val / total
    print(f"Proportion of assets with logo gt in train: {prop_train}")
    print(f"Proportion of assets with logo gt in val: {prop_val}")
    print("Sampling the assets without logo gt with the same proportions...")

    # Sample the assets without logo gt using the proportions
    df_no_logos_train, df_no_logos_val = df_no_logos.randomSplit([prop_train, prop_val], seed=42)

    # Union the DataFrames
    train_df = train_df_logos.union(df_no_logos_train)
    val_df = val_df_logos.union(df_no_logos_val)

    return train_df, val_df

def split_train_val_test(df, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1):
    fractions = [train_fraction, val_fraction, test_fraction]
    splits = df.randomSplit(fractions, seed=42)

    train_df, val_df, test_df = splits
    return train_df, val_df, test_df

def remove_logo_pseudolabels(df):
    # drop the boxes column
    df = df.drop("boxes")
    return df


def left_join_logo_gts(df, logo_gts_df):
    # join the datanet_df with the logo05_df on uri
    df = df.join(logo_gts_df, on="uri", how="left")
    df = df.dropDuplicates(["uri"])
    return df

def replace_nulls_with_empty_list(boxes):
    if boxes is None:
        return []
    else:
        return boxes

if __name__ == "__main__":


    keep_only_real_gt_logos = True
    prop_faces = 0.9
    prop_backgrounds = 0.1


    # input paths
    TOTAL_FUSION_02_PATH = os.path.join(innovation_path("mnt"), "pdacosta", "data", "total_fusion_02")
    INPUT_PATH_RETINA = os.path.join(TOTAL_FUSION_02_PATH, "retina", "annotations")
    INPUT_PATH_RETINA_EXTRA = os.path.join(TOTAL_FUSION_02_PATH, "retina", "extra", "annotations")
    INPUT_PATH_INTERNAL_FACE_DETECTOR =  os.path.join(TOTAL_FUSION_02_PATH, "internal_face_detector", "annotations")
    INPUT_PATH_DATANET = os.path.join(TOTAL_FUSION_02_PATH, "merged", "train_dataset", "complete")

    LOGO05_PATH = os.path.join(innovation_path("mnt"), "pdacosta", "data", "logo05")
    INPUT_PATH_LOGO05_TRAIN = os.path.join(LOGO05_PATH, "annotations/train/metadata/")
    INPUT_PATH_LOGO05_TEST = os.path.join(LOGO05_PATH, "annotations/test/metadata/")

    # output paths
    FUSION_PATH = os.path.join(innovation_path("mnt"), "pdacosta", "data", "fusion")
    OUTPUT_PATH = os.path.join(FUSION_PATH, "annotations")
    OUTPUT_PARQUET_PATH =  os.path.join(OUTPUT_PATH, "parquet")
    OUTPUT_PARQUET_PATH_LOGO05 =  os.path.join(OUTPUT_PATH, "parquet", "logo_05")
    OUTPUT_CSV_PATH = os.path.join(OUTPUT_PATH, "csv")


    # read parquet files from a directory in pyspark
    retina_df_1 =  read_boxes_parquet(INPUT_PATH_RETINA, spark, "retina")
    retina_df_2 =  read_boxes_parquet(INPUT_PATH_RETINA_EXTRA, spark, "retina")
    retina_df = retina_df_1.union(retina_df_2).dropDuplicates(["asset_id"])


    # # read ias boxes - ie those from the internal face detector
    ias_faces_df = read_boxes_parquet(INPUT_PATH_INTERNAL_FACE_DETECTOR, spark, "ias")

    # # merge the retina_df with the ias_df anc create faces_df
    faces_df = ias_faces_df.join(retina_df, on="asset_id", how="inner")

    # read the datanet dataframe - this only contains logo boxes
    datanet_df = spark.read.parquet(INPUT_PATH_DATANET)
    datanet_df = datanet_df.dropDuplicates(["asset_id"])
    if keep_only_real_gt_logos:

        logo05_train_df = spark.read.parquet(INPUT_PATH_LOGO05_TRAIN).select("uri", "boxes").withColumn("is_train", F.lit(1))
        logo05_test_df = spark.read.parquet(INPUT_PATH_LOGO05_TEST).select("uri", "boxes").withColumn("is_train", F.lit(0))
        # union the train and test dataframes
        logo05_df = logo05_train_df.union(logo05_test_df)

        print("datanet count:", datanet_df.count())
        print("logo05 count:", logo05_df.count())

        # merge the datanet_df with the logo05_df on uri and keep only the columns uri and boxes
        gt_logos_df = datanet_df.select("uri").join(logo05_df, on="uri", how="inner").select("uri", "boxes", "is_train")

        print("gt_logos_df (inner with datanet) count:", gt_logos_df.count())


    # merge the faces_df with the datanet_df and  cache
    df = faces_df.join(datanet_df, on="asset_id", how="inner")
    df = df.cache()

    # create a udf to transform the boxes
    transform_boxes_udf = udf(boxes_to_dict, OUTPUT_SCHEMA_BOX)

    # apply the udf to the dataframe and create the  face boxes following the yolo format
    df = df.withColumn("ias_boxes", transform_boxes_udf("ias_boxes", "width", "height", F.lit("xyxy"), F.lit("yolo"), F.lit(0)))
    df = df.withColumn("retina_boxes", transform_boxes_udf("retina_boxes", "width", "height", F.lit("xyxy"), F.lit("yolo"), F.lit(0)))

    # udf to merge the boxes
    merge_boxes_udf = udf(merge_boxes, OUTPUT_SCHEMA_BOX)

    # merge the faces together based on the iou threshold
    df =  df.withColumn("face_boxes", merge_boxes_udf("retina_boxes", "ias_boxes"))

    print("df count:", df.count())
    if keep_only_real_gt_logos:
        # remove the old boxes column and add the one with the real gt logos
        df = df.withColumnRenamed("boxes", "pseudo_boxes")
        df = left_join_logo_gts(df, gt_logos_df)
        print("df count after join with gts (should be the same):", df.count())
        replace_nulls_with_empty_list_udf = udf(replace_nulls_with_empty_list, OUTPUT_SCHEMA_BOX)
        df = df.withColumn("boxes", replace_nulls_with_empty_list_udf("boxes"))
        df = undersample_faces(df, prop=prop_faces, use_pseudo_labels=True)
        df = df.drop("pseudo_boxes")
    else:
        # # # undersample the faces
        df = undersample_faces(df, prop=prop_faces)

    # # # merge the face boxes with the logo boxes
    df = df.withColumn("boxes", F.concat(F.col("boxes"), F.col("face_boxes"))).drop("face_boxes")

    # # undersample the backgrounds
    df = undersample_backgrounds(df, prop=prop_backgrounds)

    # # # make yolo annotations
    yolo_annotations_udf = udf(yolo_annotations, StringType())
    df = df.withColumn("yolo_annotations", yolo_annotations_udf("boxes"))


    df = df.select("asset_id", "uri", "width", "height", "boxes", "box_type", "has_face", "has_logo", "yolo_annotations")
    # # create more uri columns
    df = create_uris(df)

    print(f"Splitting the dataframe into train, val and test")
    # # split the dataframe into train, val and test
    if keep_only_real_gt_logos:
        train_df, val_df = split_train_val_with_column(df, "is_train")

    else:
        train_df, val_df, test_df = split_train_val_test(df, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1)

    print(f"Number of assets in train: {train_df.count()}")
    print(f"Number of assets in val: {val_df.count()}")
    # if test_df:
    #     print(f"Number of assets in test: {test_df.count()}")

    print("Saving the dataframes...")
    
    if keep_only_real_gt_logos:
        train_df.write.mode("overwrite").parquet(OUTPUT_PARQUET_PATH_LOGO05 + "/train/")
        val_df.write.mode("overwrite").parquet(OUTPUT_PARQUET_PATH_LOGO05 + "/val/")
    else:
        # save the dataframes with the annotations
        train_df.write.mode("overwrite").parquet(OUTPUT_PARQUET_PATH + "/train/")
        val_df.write.mode("overwrite").parquet(OUTPUT_PARQUET_PATH + "/val/")
        if test_df:
            test_df.write.mode("overwrite").parquet(OUTPUT_PARQUET_PATH + "/test/")
    print(f"Saved files to:{OUTPUT_PARQUET_PATH}")
