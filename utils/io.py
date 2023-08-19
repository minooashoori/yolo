from zipfile import ZipFile
import os, re, io, boto3, shutil
import time
from tqdm import tqdm
from glob import glob

def get_s3_bucket_key(path):
    _, bucket, key, _ = re.split("s3://(.*?)/(.*)$", path)
    return bucket, key
    


def unzip_file_s3(path, target_dir):
    """
    Unzip a file from S3 to a target directory in s3.
    This is (much) slower than downloading unzipping in the to local disk.
    """
    
    
    s3 = boto3.client('s3')
    
    bucket, key = get_s3_bucket_key(path)
    print(f"Unzipping {bucket}/{key} to {target}...")
    
    print("Waiting for file to be available...")
    st = time.time()
    response = s3.get_object(Bucket=bucket, Key=key)
    zip_bytes = response['Body'].read()
    et = time.time()
    print(f"File is available. Took {et-st:.2f} seconds.")
    
    with io.BytesIO(zip_bytes) as zip_stream:
        with ZipFile(zip_stream) as zip_ref:
            
            # list of files and folders in the zip
            zip_list = zip_ref.namelist()
            for filename in tqdm(zip_list, desc="Unzipping", unit="files"):
                # check if it's a folder
                if filename.endswith('/'):
                    continue
                
                target_key = os.path.join(target_dir, filename)
                # check if the file already exists
                if s3.head_object(Bucket=bucket, Key=target_key):
                    continue
                
                file_contents = zip_ref.read(filename)
                s3.put_object(Bucket=bucket, Key=target_key, Body=file_contents)
                # print(f"Unzipped {filename} to {target_key}")

    return zip_list


def download(s3_path, local_path):
    """
    download a file from s3 to local disk
    """
    s3 = boto3.client('s3')
    
    bucket, key = get_s3_bucket_key(s3_path)
    
    # check if the file already exists
    if os.path.exists(local_path):
        print(f"File {local_path} already exists.")
        return 
    
    # check if the dir exists if not create it
    local_dir = os.path.dirname(local_path)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    print(f"Downloading from {bucket}/{key}...")
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded to {local_path}")
    



def unzip_file(path, target_dir, overwrite=False):
    """
    Unzip a file from local disk to a target directory.
    """
    
    # check if the file already exists
    if os.path.exists(target_dir):
        if overwrite:
            print(f"Directory {target_dir} already exists. Overwriting.")
            # delete the directory
            shutil.rmtree(target_dir)
        else:
            print(f"Directory {target_dir} already exists. Skipping.")
            return
    
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Unzipping {path} to {target_dir}...")
    with ZipFile(path) as zip_ref:
        zip_ref.extractall(target_dir)
    
    print(f"Unzipped {path} to {target_dir}")
    
    
def list_files(dir, ext):
    """
    List all images in a directory.
    """
    return glob(os.path.join(dir, "**" ,f"*.{ext}"), recursive=True)


def mv_to_dir(input_dir, target_dir, ext, rename: bool):
    """
    Move all files with a given extension to a folder.
    """
    files = list_files(input_dir, ext)
    
    print(f"Moving {len(files)} files to {target_dir}...")
    # create the target dir
    os.makedirs(target_dir, exist_ok=True)
    
    # we have to rename the files to avoid name collisions
    num_digits = len(str(len(files)))
    # we will create files with the format 000000.ext where the number of zeros is the number of files
    i = 0
    old_new_names_map = {}
    for file_ in files:
        # get the new name
        new_name = f"logodet3k_{str(i).zfill(num_digits)}.{ext}"
        # move the file
        if rename:
            new_filepath = os.path.join(target_dir, new_name)
            shutil.move(file_, new_filepath)
        else:
            # if we don't rename we need to keep the folder structure
            # so we need to remove the input_dir from the file path
            # and add it to the target_dir
            path_to_file = file_.replace(input_dir, "")
            new_filepath = os.path.join(target_dir, path_to_file.lstrip(os.path.sep))
            # new_filepath = target_dir + file_.replace(input_dir, "")
            print("new_filepath", new_filepath)
            # if the dir doesn't exist create it
            os.makedirs(os.path.dirname(new_filepath), exist_ok=True)
            
            shutil.move(file_, new_filepath)
        old_new_names_map[file_] = new_filepath
        i += 1
    
    print(f"Moved {len(files)} files to {target_dir}")
    
    return old_new_names_map
    

def compress_dir(input_dir):
    """
    Compress a directory to a zip file.
    """
    # get the name of the dir
    dir_name = os.path.basename(input_dir)
    # get the parent dir
    parent_dir = os.path.dirname(input_dir)
    # get the path to the zip file
    zip_path = os.path.join(parent_dir, dir_name)
    
    print(f"Compressing {input_dir} to {zip_path}...")
    shutil.make_archive(zip_path, 'zip', input_dir)
    print(f"Compressed {input_dir} to {zip_path}")
    
    return zip_path + ".zip"





if __name__ == "__main__":
    
    os.environ["AWS_PROFILE"] = "saml"
    
    # s3_path = "s3://mls.us-east-1.innovation/pdacosta/data/logodet_3k/zip/LogoDet-3K.zip"
    # target = "pdacosta/data/logodet_3k/unzip"
    
    # download(s3_path, "/home/ec2-user/dev/data/logodet3k/LogoDet-3K.zip")
    
    
    