from zipfile import ZipFile
import os, re, io, boto3, shutil
import time
from tqdm import tqdm
from glob import glob

def get_s3_bucket_key(path):
    _, bucket, key, _ = re.split("s3://(.*?)/(.*)$", path)
    return bucket, key

#

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


def download(s3_path, local_filepath):
    """
    Download a file from S3 to a local directory.
    """
    s3 = boto3.client('s3')

    bucket, key = get_s3_bucket_key(s3_path)

    # if we mistaknely pass a directory as the local path
    # we need to add the filename to the local path
    if os.path.isdir(local_filepath):
        local_filepath = os.path.join(local_filepath, os.path.basename(key))

    # Create the local directory if it doesn't exist
    local_dir = os.path.dirname(local_filepath)
    os.makedirs(local_dir, exist_ok=True)

    # if the file already exists skip
    if os.path.exists(local_filepath):
        print(f"File {local_filepath} already exists. Skipping.")
        return

    print(f"Downloading from {bucket}/{key}... to {local_filepath}")
    s3.download_file(bucket, key, local_filepath)
    print(f"Downloaded to {local_filepath}")




def unzip_file(path, target_dir=None, delete_zip=False):
    """
    Unzip a file from local disk to a target directory.
    """

    if target_dir is None:
        target_dir = os.path.dirname(path)

    # check if the file already exists
    if os.path.exists(target_dir):
        print(f"Directory to extract the files {target_dir} already exists. This might overwrite existing files.")

    os.makedirs(target_dir, exist_ok=True)

    print(f"Unzipping {path} to {target_dir}...")
    with ZipFile(path) as zip_ref:
        zip_ref.extractall(target_dir)

    print(f"Unzipped {path} to {target_dir}")
    if delete_zip:
        os.remove(path)
        print(f"Deleted {path}")


def list_files(dir, ext):
    """
    List all images in a directory.
    """
    return glob(os.path.join(dir, "**" ,f"*.{ext}"), recursive=True)


def mv_to_dir(input_dir: str, target_dir: str, ext: str, rename: bool, keep_dir_structure: bool = False, prefix: str = None):
    """
    Move all files with a given extension to a folder.
    """
    files = list_files(input_dir, ext)
    num_files = len(files)

    print(f"Moving {num_files} files to {target_dir}...")

    # Create the target directory
    os.makedirs(target_dir, exist_ok=True)

    # Determine the number of digits needed for renaming
    num_digits = len(str(num_files))

    old_new_names_map = {}

    # raise warning if we have prefix with rename=False
    if not rename and prefix:
        print("Warning: rename=False and prefix is not None. Prefix will be ignored.")

    for i, file_ in enumerate(files):
        # Determine the new name for the file
        if rename:
            new_name = f"{prefix}_{str(i).zfill(num_digits)}.{ext}" if prefix else f"{str(i).zfill(num_digits)}.{ext}"
        else:
            new_name = os.path.basename(file_)


        # if we want to keep the directory structure then we need to remove the input_dir from the path
        if keep_dir_structure:
            path_to_file = file_.replace(input_dir, "")
            if rename:
                # change the name of the file in path_to_file
                path_to_file = path_to_file.replace(os.path.basename(file_), new_name)

            new_filepath = os.path.join(target_dir, path_to_file.lstrip(os.path.sep))
        else:
            new_filepath = os.path.join(target_dir, new_name)


        # Create the necessary directories
        os.makedirs(os.path.dirname(new_filepath), exist_ok=True)

        # Move the file
        shutil.move(file_, new_filepath)
        old_new_names_map[file_] = new_filepath

    print(f"Moved {num_files} files to {target_dir}")
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


