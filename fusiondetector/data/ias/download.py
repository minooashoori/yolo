import boto3
import s3fs
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from utils.io import get_s3_bucket_key
import os
from io import BytesIO 
import sys
sys.path.append("/home/ec2-user/dev/img2dataset")
import img2dataset as i2d

os.environ['AWS_PROFILE'] = 'saml'


# def read_parquet_from_s3(path):
#     s3 = boto3.client('s3')
#     bucket, key = get_s3_bucket_key(path)
#     print(f"Reading from {bucket}, {key}...")
#     obj = s3.get_object(Bucket=bucket, Key=key)
#     parquet_content = obj['Body'].read()
#     buffer = BytesIO(parquet_content)
#     return pq.read_parquet(buffer).to_pandas()

def read_parquet_s3fs(path):
    fs = s3fs.S3FileSystem()
    bucket_name, folder_path = get_s3_bucket_key(path)
    parquet_files = fs.glob(f'{bucket_name}/{folder_path}/*.parquet')

    dfs = []

    for file in parquet_files:
        with fs.open(file) as f:
            df = pd.read_parquet(f)
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)



def save_s3_uris(df, path):
    """
    Saves the s3 uris in a txt file in the local disk without the commas
    """
    df['s3_uri'].to_csv(path, index=False, header=False, sep='\n')
    

# df = read_parquet_s3fs('s3://mls.us-east-1.innovation/pdacosta/data/total_fusion_02/merged/train_dataset/undersampled/')
# print(f"Shape: {df.shape}")
# save_s3_uris(df, '/home/ec2-user/dev/data/ias/s3_uris.txt')


if __name__ == '__main__':
    output_dir = '/home/ec2-user/dev/data/ias/dataset/'
    
    i2d.download(
        processes_count=16,
        thread_count=32,
        url_list="/home/ec2-user/dev/data/ias/s3_uris.txt",
        output_folder=output_dir,
        output_format="webdataset",
        input_format="txt",
        enable_wandb=False,
        number_sample_per_shard=10000,
        distributor="multiprocessing",
        resize_mode="no"    
    )