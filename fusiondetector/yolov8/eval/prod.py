import os
import s3fs
import pandas as pd


def read_csv(path):
    return pd.read_csv(path)

def download_csv(path, save_path):
    fs = s3fs.S3FileSystem()
    fs.download(path, save_path)
    return save_path

def make_annotation_files(df, save_path, column:str, conf=None):
    # each row is a file, we can use asset_id as filename
    os.makedirs(save_path, exist_ok=True)
    for row in df.itertuples():
        # check if asset_id in in the list of columns
        if "asset_id" in df.columns:
            asset_id = row.asset_id
        elif "asset" in df.columns:
            asset_id = row.asset
        # image filename without extension
        uri = row.uri
        # there are some nan values in uri
        if pd.isna(uri):
            continue
        filename = os.path.splitext(os.path.basename(uri))[0]
        col = getattr(row, column)
        with open(os.path.join(save_path, f"{filename}.txt"), "w") as f:
            if isinstance(col, str):
                if conf:
                    col = filter_conf(col, conf)
                f.write(col)
            else:
                print(filename, col)


def filter_conf(annotation, conf):
    # annotation is a string with the format:
    # class conf x y w h
    # we want to filter by conf
    # return a string with the same format
    lines = annotation.split("\n")
    filtered_lines = []
    for line in lines:
        if line:
            line = line.split(" ")
            if float(line[1]) >= conf:
                filtered_lines.append(" ".join(line))
    return "\n".join(filtered_lines)


# test_df_path = "s3://mls.us-east-1.innovation/pdacosta/data/logo05/annotations/gts_preds/xywh/test/part-00000-tid-8483272553848330383-f70032ca-4fbb-4013-9f7a-f3e3c2182c3f-2325-1-c000.csv"
# local_test_df_path =  "/home/ec2-user/dev/data/logo05/annotations/gts_preds/test.csv"

# # download_csv(test_df_path, local_test_df_path)

# test_df = read_csv(local_test_df_path)

# # make_annotation_files(test_df, "/home/ec2-user/dev/data/logo05/annotations/gts_preds/gts", "gt_boxes_abs")
# make_annotation_files(test_df, "/home/ec2-user/dev/data/logo05/annotations/gts_preds/preds", "pred_boxes_abs", conf=0.35)


# widerface_val_df_path = "s3://mls.us-east-1.innovation/pdacosta/data/wider_face/gts_preds/csv/xywh/part-00000-tid-5917647135633358923-4f82d01f-bd6f-49e7-a9e4-2890bab3555b-151-1-c000.csv"
local_widerface_val_df_path = "/home/ec2-user/dev/data/widerface/gts_preds/csv/xywh/val.csv"
# download_csv(widerface_val_df_path, local_widerface_val_df_path)
widerface_val_df = read_csv(local_widerface_val_df_path)
# make_annotation_files(widerface_val_df, "/home/ec2-user/dev/data/widerface/gts_preds/gts", "gt_xywh")
make_annotation_files(widerface_val_df, "/home/ec2-user/dev/data/widerface/gts_preds/pred", "det_xywh")
