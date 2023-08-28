from utils.boxes import transform_box, iou_yolo, transf_any_box, fix_bounds_relative, relative
from pyspark.sql.functions import udf
import pyspark.sql.functions as F
from pyspark.sql.types import LongType, ArrayType, StructType, StructField, DoubleType
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

output_schema_dict = ArrayType(StructType([
    StructField("category", LongType(), True),
    StructField("x", DoubleType(), True),
    StructField("y", DoubleType(), True),
    StructField("width", DoubleType(), True),
    StructField("height", DoubleType(), True)
]))


def boxes_to_dict(boxes, width, height, input_type, output_type, class_id=0):
    if len(boxes) == 0:
        return []
    transformed_boxes = []
    if isinstance(boxes, str):
        boxes = ast.literal_eval(boxes)
    for box in boxes:
        transf_box =  relative(width=width, height=height, box=box) # just in case we have absolute coordinates
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




def merge_boxes(retina_boxes, ias_boxes, iou_threshold=0.25):

    if len(retina_boxes) == 0:
        return ias_boxes
    if len(ias_boxes) == 0:
        return retina_boxes

    # we check retina boxes against ias boxes, if there is an overlap we discard the retina box, else we keep it
    include_boxes = []
    for retina_box in retina_boxes:
        x_r, y_r, w_r, h_r = retina_box["x"], retina_box["y"], retina_box["width"], retina_box["height"]
        r_box = [x_r, y_r, w_r, h_r]
        overlap = False
        for ias_box in ias_boxes:
            x_i, y_i, w_i, h_i = ias_box["x"], ias_box["y"], ias_box["width"], ias_box["height"]
            i_box = [x_i, y_i, w_i, h_i]
            # check if there is an overlap with at least one ias box
            if iou_yolo(r_box, i_box) > iou_threshold:
                overlap = True
                break
        if not overlap:
            include_boxes.append(retina_box)

    return include_boxes + ias_boxes



def read_boxes_parquet(path, spark, alg_type:str):
    df = spark.read.parquet(path)
    df = df.drop("img_bytes").select("asset_id", "boxes")
    df = df.repartition(1000)
    df = df.withColumnRenamed("boxes", f"{alg_type}_boxes")
    df = df.dropDuplicates(["asset_id"])
    return df


def make_patch_format(box, width, height):
    x_c, y_c, w, h = box
    x1 = x_c - w/2
    y1 = y_c - h/2

    x1 = x1 * width
    y1 = y1 * height
    w = w * width
    h = h * height
    return x1, y1, w, h


def plot_boxes(uri, retina_boxes, ias_boxes, face_boxes, width, height, retina_color="r", ias_color="b", face_color="g"):

    uri = uri.replace("mls.godfather", "/dbfs/mnt/mls.godfather")

    # open the image
    img = Image.open(uri)

    # we need to make the figure and the axes
    fig, ax = plt.subplots(1)

    # add the image to the axes
    ax.imshow(img)

    # boxes come in the yolo format, x_center, y_center, width, height
    # we need to transform them to the xyxy format
    if retina_boxes:
        for retina_box in retina_boxes:
            x1, y1, w, h = make_patch_format(retina_box, width, height)
            # create a rectangle patch
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=retina_color, facecolor='none')
            # add the patch to the axes
            ax.add_patch(rect)

    if ias_boxes:
        for ias_box in ias_boxes:
            x1, y1, w, h = make_patch_format(ias_box, width, height)

            # create a rectangle patch
            rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor=ias_color, facecolor='none')
            # add the patch to the axes
            ax.add_patch(rect)

    if face_boxes:
        for face_box in face_boxes:
            x1, y1, w, h = make_patch_format(face_box, width, height)

            # create a rectangle patch
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=face_color, facecolor='none')
            # add the patch to the axes
            ax.add_patch(rect)


    plt.show()



if __name__ == "__main__":

    # read parquet files from a directory in pyspark
    retina_df_1 = read_boxes_parquet("/mnt/innovation/pdacosta/data/total_fusion_02/retina/annotations/", spark, "retina")
    retina_df_2 = read_boxes_parquet("/mnt/innovation/pdacosta/data/total_fusion_02/retina/extra/annotations/", spark, "retina")
    retina_df = retina_df_1.union(retina_df_2).dropDuplicates(["asset_id"])


    # read ias boxes - ie those from the internal face detector
    ias_df = read_boxes_parquet("/mnt/innovation/pdacosta/data/total_fusion_02/internal_face_detector/annotations/", spark, "ias")

    # merge the retina_df with the ias_df anc create faces_df
    faces_df = ias_df.join(retina_df, on="asset_id", how="inner")

    # read the datanet dataframe - this only contains logo boxes
    datanet_df = spark.read.parquet("/mnt/innovation/pdacosta/data/total_fusion_02/merged/train_dataset/complete/")
    datanet_df = datanet_df.dropDuplicates(["asset_id"])

    # merge the faces_df with the datanet_df and  cache
    df = faces_df.join(datanet_df, on="asset_id", how="inner")
    df = df.cache()

    # create a udf to transform the boxes
    transform_boxes_udf = udf(boxes_to_dict, output_schema_dict)

    # apply the udf to the dataframe and create the  face boxes following the yolo format
    df = df.withColumn("ias_boxes", transform_boxes_udf("ias_boxes", "width", "height", F.lit("xyxy"), F.lit("yolo"), F.lit(0)))
    df = df.withColumn("retina_boxes", transform_boxes_udf("retina_boxes", "width", "height", F.lit("xyxy"), F.lit("yolo"), F.lit(0)))

    # udf to merge the boxes
    merge_boxes_udf = udf(merge_boxes, output_schema_dict)

    # merge the face together based on the iou threshold
    df =  df.withColumn("face_boxes", merge_boxes_udf("retina_boxes", "ias_boxes"))



    # add a variable to indicate if the asset has a face box
    df = df.withColumn("has_face", F.when(F.size(F.col("face_boxes")) > 0, 1).otherwise(0))
    # same for number of boxes
    df = df.withColumn("n_face_boxes", F.size(F.col("face_boxes")))

    # similarly add a variable to indicate if the asset has a logo box
    df = df.withColumn("has_logo", F.when(F.size(F.col("boxes")) > 0, 1).otherwise(0))

    # add a variable to indicate the number of logo boxes
    df = df.withColumn("n_logo_boxes", F.size(F.col("boxes")))

    # Calculate the number of assets with face boxes only
    n_assets_with_face = df.filter((F.col("has_face") == 1) & (F.col("has_logo") == 0)).count()

    # Calculate the number of assets with logo boxes only
    n_assets_with_logo = df.filter((F.col("has_face") == 0) & (F.col("has_logo") == 1)).count()

    # we will undersample the assets with faces to match the assets with logos
    print("Undersample the assets with faces to match the assets with logos")
    fraction_face_logo = n_assets_with_logo/n_assets_with_face
    fraction_face_logo = min(fraction_face_logo, 1.0)

    df_face = df.filter((F.col("has_face") == 1) & (F.col("has_logo")== 0)).sample(False, fraction_face_logo, seed=42)

    # remove the assets with faces and no logos from df, ie keep only the assets with logos, or with both or neither
    df_neither = df.filter((F.col("has_face") == 0) & (F.col("has_logo")== 0))
    df_both = df.filter((F.col("has_face") == 1) & (F.col("has_logo")== 1))
    df_logo = df.filter((F.col("has_face") == 0) & (F.col("has_logo")== 1))

    df = df_neither.union(df_both).union(df_logo).union(df_face)

    print(f"Number of assets after undersampling faces: {df.count()}")

    # add the face boxes to the boxes column and remove the face_boxes column
    df = df.withColumn("boxes", F.concat(F.col("boxes"), F.col("face_boxes"))).drop("face_boxes")

    df = df.select("asset_id", "uri", "width", "height", "boxes", "box_type", "has_face", "n_face_boxes", "has_logo", "n_logo_boxes")

    print("Number of assets after merging the boxes: {}".format(df.count()))

    # split the rows that don't have a box
    df_no_boxes = df.filter(F.size(F.col("boxes")) == 0)

    # split the rows that have a box
    df_boxes = df.filter(F.size(F.col("boxes")) > 0)

    print("Number of assets with boxes: {}".format(df_boxes.count()))
    print("Number of assets without boxes: {}".format(df_no_boxes.count()))

    # sample the assets without boxes to be max 2%*n_assets_with_boxes
    fraction = (0.02*df_boxes.count())/df_no_boxes.count()
    fraction = min(fraction, 1.0)

    print(f"Keep {fraction*100}% of the assets without boxes")

    df_no_boxes = df_no_boxes.sample(False, fraction, seed=42)

    # stack the two dataframes again
    df = df_boxes.union(df_no_boxes)
    print("Final number of assets: {}".format(df.count()))


    #save the dataframe to parquet
    df.write.mode("overwrite").parquet("/mnt/innovation/pdacosta/data/total_fusion_02/merged/train_dataset/undersampled/")

    # show the statistics by has_face and has_logo
    df.groupBy("has_face", "has_logo").count().show(truncate=False)

    # show the statistics by n_face_boxes, n_logo_boxes
    df.groupBy("n_face_boxes").count().show(truncate=False)
    df.groupBy("n_logo_boxes").count().show(truncate=False)


    # # select a sample of 100 assets in which both algorithms detected faces
    # sample_size = 200/df.count()
    # df_sample = df.filter((F.size(F.col("face_boxes")) > 0) & (F.size(F.col("boxes")) > 0)).sample(False, sample_size, seed=42)

    # df_sample = df_sample.select("asset_id", "width", "height", "ias_boxes", "retina_boxes", "uri", "face_boxes", "boxes")

    # # collect the sample and transform to a python dictionary
    # sample = df_sample.collect()

    # # create a dictionary with the asset_id as key and the uri as value
    # asset_dict = {row["asset_id"]:
    #     {"uri": row["uri"],
    #      "ias_boxes": [ [r["x"], r["y"], r["width"], r["height"] ] for r in row["ias_boxes"]],
    #      "retina_boxes": [[r["x"], r["y"], r["width"], r["height"]] for r in row["retina_boxes"]],
    #      "face_boxes": [[r["x"], r["y"], r["width"], r["height"]] for r in row["face_boxes"]],
    #      "boxes": [[r["x"], r["y"], r["width"], r["height"]] for r in row["boxes"]],
    #      "width": row["width"],
    #      "height": row["height"]} for row in sample}

    # # plot the images with the boxes
    # for asset_id, asset in asset_dict.items():
    #     uri = asset["uri"]
    #     ias_boxes = asset["ias_boxes"]
    #     retina_boxes = asset["retina_boxes"]
    #     face_boxes = asset["face_boxes"]
    #     boxes = asset["boxes"]
    #     width = asset["width"]
    #     height = asset["height"]
    #     plot_boxes(uri, boxes, None, face_boxes, width, height)
