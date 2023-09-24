import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
import pandas as pd
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import numpy as np
import os
from retinaface import RetinaFace

os.environ["AWS_PROFILE"]="saml"

# # read all the .parquet files from the folder and creae a table
# parquet_path = "s3://mls.us-east-1.innovation/pdacosta/data/total_fusion_02/train_dataset/extra/output/"

s3 = s3fs.S3FileSystem()

# parquet_files = s3.ls(parquet_path)
# #add s3:// to the file names
# parquet_files = ["s3://" + file for file in parquet_files]
# # remove the folder name from the list and _SUCCESS
# parquet_files = [file for file in parquet_files if file != parquet_path and file != parquet_path + "_SUCCESS"]

# dfs = [pd.read_parquet(location) for location in parquet_files]

# # Concatenate the DataFrames if needed
# df = pd.concat(dfs, ignore_index=True)

# # extract only the uri column and save it to a list
# uri_list = df["uri"].tolist()
# asset_ids = df["asset_id"].tolist()
# img_paths = ["s3://" + path for path in uri_list]


df = pd.read_csv('/home/ec2-user/dev/data/portal/image_urls.csv')
uri_list = df["s3_uri"].tolist()
img_paths = uri_list
n_images = len(img_paths)

print("Total number of images: {}".format(n_images))

# Check if there's a progress file, and if not, start from the beginning
progress_filename = "progress.txt"
if os.path.exists(progress_filename):
    with open(progress_filename, "r") as progress_file:
        last_processed_index = int(progress_file.read())
else:
    last_processed_index = 0


def save_file_update_progress(data, idx, format="csv"):
    # parquet file with columns: asset_id, img_bytes, boxes
    df = pd.DataFrame(data)
    if format == "parquet":
        table = pa.Table.from_pandas(df)
        parquet_filename = f"retina_{idx}.parquet"
        pq.write_table(table, parquet_filename)
    elif format == "csv":
        csv_filename = f"retina_{idx}.csv"
        df.to_csv(csv_filename)

    # Update the progress file with the current index
    print("Saving progress up to {}...".format(idx))

    with open(progress_filename, "w") as progress_file:
        progress_file.write(str(idx))

# # we are ready to start with the inference




data_list = []
pbar = tqdm(total=len(img_paths), initial=last_processed_index ,desc="Processing images")

for i in range(last_processed_index, n_images):


    img_path = img_paths[i]

    try:
        with s3.open(img_path, "rb") as f:
            img_bytes = f.read()
            img  = Image.open(BytesIO(img_bytes))
            img_array = np.array(img)
    except:
        continue


    # if the image doesn't have 3 channels or is empty, skip it
    if len(img_array.shape) != 3 or img_array.shape[2] != 3 or img_array.shape[0] == 0 or img_array.shape[1] == 0:
        continue

    detection = RetinaFace.detect_faces(img_array, threshold=0.7)

    boxes = []
    if isinstance(detection, dict):
        for k, v in detection.items():
            box = v["facial_area"]
            boxes.append(box)

    data  = {
        "s3_uri": img_path,
        "boxes": boxes
    }

    data_list.append(data)


    if (i + 1) % 10000 == 0:
        save_file_update_progress(data_list, i+1, format="csv")
        data_list = []

    pbar.update(1)

# save the last batch of data
if data_list:
    save_file_update_progress(data_list, n_images, format="csv")
