import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"img2dataset"))

import img2dataset as i2d
import webdataset as wds
import tarfile
import shutil
import glob

from utils.boxes import plot_boxes
from torch.utils.data import DataLoader

os.environ['AWS_PROFILE'] = 'saml'

class ImageDataset:

    """`Dataset` is the base class for all datasets. It provides the following
    methods:
    download: Downloads the dataset from s3 and saves it to disk.
    format: Formats the dataset into a standard format used by yolo models.
    wds: creates a webdataset from the dataset.
    """

    def __init__(self,
                name:str,
                input_path: str) -> None:
        self.name = name
        self.input_path = input_path

    def download(self,
                path:str) -> None:
        """Downloads the dataset from s3 and saves it to disk.
        """
        raise NotImplementedError

    def format(self,
                path:str) -> None:
        """Formats the dataset into a standard format used by yolo models.
        """
        raise NotImplementedError

    def wds(self,
            path:str) -> None:
        """Creates a webdataset from the dataset.
        """
        raise NotImplementedError

    @staticmethod
    def _dataset_exists(path:str):
        """
        Check if the dataset (files) exist in the local path. If it does, return True, else return False.
        """
        #check if there are any files in the path, if so, return True
        if os.path.exists(path):
            if os.listdir(path):
                return True
            else:
                return False
        else:
            return False



class TotalFusionDataset(ImageDataset):

    def __init__(self, name: str, input_path: str) -> None:
        super().__init__(name, input_path)
        self.dataset_path = None

    def download(self,
                input_format: str,
                output_folder: str = None,
                output_format="webdataset",
                overwrite=False) -> None:
        """
        Downloads the dataset from s3 and saves it to disk. Note that we can also save directly to s3 if the output_folder is an s3 path.
        By default, the dataset is downloaded in webdataset format.
        """

        if not output_folder:
            home_path = os.path.expanduser("~")
            output_folder = os.path.join(home_path, "dev/data/totalfusion/dataset", self.name)

        if self._dataset_exists(output_folder):
            if overwrite:
                print(f"Dataset already exists in {output_folder}. Overwriting.")
                # delete the existing dataset
                shutil.rmtree(output_folder)
            else:
                self.dataset_path = output_folder
                print(f"Dataset already exists in {output_folder}. Skipping download.")
                return

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


        i2d.download(
            processes_count=16,
            thread_count=32,
            url_list=self.input_path,
            output_folder=output_folder,
            output_format=output_format,
            input_format=input_format,
            number_sample_per_shard=10000,
            distributor="multiprocessing",
            resize_mode="keep_ratio_largest",
            resize_only_if_bigger=True,
            enable_wandb=True,
            encode_quality=90,
            skip_reencode=True,
            image_size=416,
            url_col="s3_uri",
            caption_col="yolo_annotations",
        )

        self.dataset_path = output_folder

    def format(self,
            output_folder:str = None,
            dataset_path:str = None,
            output_format:str = 'yolo') -> None:
        """
        Formats the dataset into a standard format used by yolo models.
        """

        # we will use webdataset to read the dataset, and then format it into yolo format - ideally we should be able to use just webdataset in the future.
        if dataset_path:
            self.dataset_path = dataset_path
        if not self.dataset_path:
            raise ValueError("Dataset path not specified. Please specify the path to the dataset.")

        if not output_folder:
            home_path = os.path.expanduser("~")
            output_folder = os.path.join(home_path, "dev/data/totalfusion/yolo")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # get the list of tar files
        dataset_path = os.path.join(self.dataset_path, "*.tar")
        tarball_files = glob.glob(dataset_path)
        
        for tarball_file in tarball_files:
            print(f"Processing {tarball_file}...")
            self._process_tar(tarball_file, output_folder)
            print(f"Finished processing {tarball_file}.")

    def _process_tar(self, tar_path, output_folder):

        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    if member.name.endswith(".jpg"):
                        self._copy_file(tar, member, output_folder, "images")
                    elif member.name.endswith(".txt"):
                        self._copy_file(tar, member, output_folder, "labels")

    def _copy_file(self, tar, member, output_folder, file_type):

        assert file_type in ["images", "labels"], "file_type must be one of images or labels"

        filename = os.path.basename(member.name)
        destination_path = os.path.join(output_folder, file_type, self.name, filename)

        # make sure the destination folder exists
        if not os.path.exists(os.path.dirname(destination_path)):
            os.makedirs(os.path.dirname(destination_path))

        with tar.extractfile(member) as src_file:
            with open(destination_path, "wb") as dst_file:
                shutil.copyfileobj(src_file, dst_file)


class LogoDet3KDataset(ImageDataset):
    
    def __init__(self, name: str, input_path: str) -> None:
        super().__init__(name, input_path)



if  __name__ == "__main__":

    tfusion_val = TotalFusionDataset(
        name="val",
        input_path="s3://mls.us-east-1.innovation/pdacosta/data/total_fusion_02/annotations/parquet/val/",
        )
    tfusion_val.download(input_format="parquet",
                        overwrite=False)
    # tfusion_val.format()
    
    tfusion_train = TotalFusionDataset(
        name="train",
        input_path="s3://mls.us-east-1.innovation/pdacosta/data/total_fusion_02/annotations/parquet/train/",
        )
    tfusion_train.download(input_format="parquet",
                        overwrite=False)
    
    tfusion_train.format()
    
    home_path = os.path.expanduser("~")
    dataset_path = os.path.join(home_path, "dev/data/totalfusion/dataset", "val", "*.tar")
    import glob
    tarball_files = glob.glob(dataset_path)
    webdata = wds.WebDataset(tarball_files).decode("rgb").to_tuple("jpg", "txt")
    # webdata = webdata.batched(32)
    i=0
    for img, txt in webdata:
        if i == 19:
            break
        i+=1

    import matplotlib.pyplot as plt
    #save the image to disk
    plot_boxes(img, yolo_annotation=txt, save=True)
    
    plt.imsave("test.jpg", img)
    # print(txt)
