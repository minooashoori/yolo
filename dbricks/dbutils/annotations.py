from dbutils.boxes import *


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


def read_boxes_parquet(path, spark, type_:str):
    df = spark.read.parquet(path)
    # df = df.select("asset_id", "boxes", "width", "height")
    # df = df.withColumnRenamed("boxes", f"{type_}_boxes")
    # df = df.dropDuplicates(["asset_id"])
    return df