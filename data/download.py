import boto3
from utils.io import get_s3_bucket_key
import os

class DownloadDataset:
    def __init__(self, 
                 s3_path, 
                 local_path):
        self.s3_path = s3_path
        self.local_path = local_path
        self.s3 = boto3.client('s3')
        self.bucket, self.key = get_s3_bucket_key(s3_path)
    
    def _list_files(self):
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=self.key)
        return response['Contents']

    def _make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    def _download_file(self, key, local_path):
        local_file_path = os.path.join(local_path, os.path.basename(key)) 
        print("Downloading", key, "to", local_file_path)
        self.s3.download_file(self.bucket, key, local_file_path)
    
    def _download_files(self, files):
        for file in files:
            key = file['Key']
            print("Downloading", key)
            local_file_path = os.path.join(self.local_path, os.path.basename(key)) 
            self._download_file(key, local_file_path)
        
    def download(self):
        raise NotImplementedError
            

        
class DownloadLogoDet3k(DownloadDataset):
    def __init__(self, 
                 s3_path, 
                 local_path):
        super().__init__(s3_path, local_path)
    
    def download(self):
        files = self._list_files()
        self._make_dir(self.local_path)
        self._download_files(files)
        
if __name__ == "__main__":
    os.environ["AWS_PROFILE"] = "saml"
    
    s3_path = "s3://mls.us-east-1.innovation/pdacosta/data/logodet_3k/annotations/csv/"
    local_path = "/home/ec2-user/data/logodet3k/annotations/"
    downloader = DownloadLogoDet3k(s3_path, local_path)
    downloader.download()