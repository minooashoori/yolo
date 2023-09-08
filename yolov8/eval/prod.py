import os
import s3fs
import pandas as pd


def read_csv(path):
    return pd.read_csv(path)

def download_csv(path, save_path):
    fs = s3fs.S3FileSystem()
    fs.download(path, save_path)
    return save_path

def make_annotation_files(df, save_path, column:str):
    # each row is a file, we can use asset_id as filename
    os.makedirs(save_path, exist_ok=True)
    for row in df.itertuples():
        asset_id = row.asset_id
        # image filename without extension
        uri = row.uri
        # there are some nan values in uri
        if pd.isna(uri):
            continue
        filename = os.path.splitext(os.path.basename(uri))[0]
        col = getattr(row, column)
        with open(os.path.join(save_path, f"{filename}.txt"), "w") as f:
            #  if  col is not string print
            if isinstance(col, str):
                f.write(col)
            else:
                print(filename, col)


test_df_path = "s3://mls.us-east-1.innovation/pdacosta/data/logo05/annotations/gts_preds/xywh/test/part-00000-tid-8483272553848330383-f70032ca-4fbb-4013-9f7a-f3e3c2182c3f-2325-1-c000.csv"
local_test_df_path =  "/home/ec2-user/dev/data/logo05/annotations/gts_preds/test.csv"

download_csv(test_df_path, local_test_df_path)

test_df = read_csv(local_test_df_path)

make_annotation_files(test_df, "/home/ec2-user/dev/data/logo05/annotations/gts_preds/gts", "gt_boxes_abs")
make_annotation_files(test_df, "/home/ec2-user/dev/data/logo05/annotations/gts_preds/preds", "pred_boxes_abs")
