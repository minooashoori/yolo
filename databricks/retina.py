from preprocess.boxes import transform_box
from pyspark.sql.functions import udf
import pyspark.sql.functions as F
from pyspark.sql.types import LongType, ArrayType, StructType, StructField, DoubleType


output_schema = ArrayType(StructType([
    StructField("category", LongType(), True),
    StructField("x", DoubleType(), True),
    StructField("y", DoubleType(), True),
    StructField("width", DoubleType(), True),
    StructField("height", DoubleType(), True)
]))


def transform_boxes(boxes, width, height, box_type="yolo", is_relative=True):
    image_size = (width, height)
    if len(boxes) == 0:
        return []
    transformed_boxes = []
    for box in boxes:
        transf_box = transform_box(box, image_size, box_type, is_relative)
        if box_type in ["yolo", "xywh"]:
            x, y, w, h = transf_box
            box_dict = {"category": 0, "x": x, "y": y, "width": w, "height": h}
        elif box_type == "xyxy":
            x1, y1, x2, y2 = box
            box_dict = {"category": 0, "x": x1, "y": y1, "x2": x2, "y2": y2}
        else:
            raise ValueError("Box type not supported")
        
        transformed_boxes.append(box_dict)
    return transformed_boxes


if __name__ == "__main__":

    # read parquet files from a directory in pyspark
    retina_df = spark.read.parquet("/mnt/innovation/pdacosta/data/total_fusion_02/retina/annotations/")
    retina_df = retina_df.drop("img_bytes")
    retina_df = retina_df.repartition(1000)
    # change the name of the column boxes to be face_boxes,
    retina_df = retina_df.withColumnRenamed("boxes", "face_boxes") 
    
    print("Number of assets in retina: {}".format(retina_df.count()))

    datanet_df = spark.read.parquet("/mnt/innovation/pdacosta/data/total_fusion_02/merged/train_dataset/metadata/")
    datanet_df = datanet_df.repartition(1000)
    
    print("Number of assets in datanet: {}".format(datanet_df.count()))

    # merge the dataframe with the previous dataframe containing the image size
    df = retina_df.join(datanet_df, on="asset_id", how="inner")
    #cache the dataframe to make it faster later on
    df.cache()

    # create a udf to apply the function to the dataframe
    transform_boxes_udf = udf(transform_boxes, output_schema)

    # apply the udf to the dataframe
    df = df.withColumn("face_boxes", transform_boxes_udf("face_boxes", "width", "height"))

    # add the face boxes to the boxes column and remove the face_boxes column
    df = df.withColumn("boxes", F.concat(F.col("boxes"), F.col("face_boxes"))).drop("face_boxes")

    # split the rows that don't have a box
    df_no_boxes = df.filter(F.size(F.col("boxes")) == 0)
    
    # split the rows that have a box
    df_boxes = df.filter(F.size(F.col("boxes")) > 0)
    n_assets_with_boxes = df_boxes.count()
    print("Number of assets with boxes: {}".format(n_assets_with_boxes))
    print("Number of assets without boxes: {}".format(df_no_boxes.count()))

    # sample the assets without boxes to be max 5%*n_assets_with_boxes
    fraction = (0.05*n_assets_with_boxes)/df_no_boxes.count()
    fraction = min(fraction, 1.0)

    print(f"Keep {fraction*100}% of the assets without boxes")
    
    df_no_boxes = df_no_boxes.sample(False, fraction, seed=42)

    # stack the two dataframes again
    df = df_boxes.union(df_no_boxes)
    print("Final number of assets: {}".format(df.count()))

    #save the dataframe to parquet
    df.write.mode("overwrite").parquet("/mnt/innovation/pdacosta/data/total_fusion_02/merged/train_dataset/metadata_with_faces/")

