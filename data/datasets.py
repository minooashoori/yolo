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
    format_yolo: Formats the dataset into a standard format used by yolo (ultralytics) models.
    wds: creates a webdataset from the dataset.
    """

    def __init__(self,
                split:str,
                input_path: str) -> None:
        self.split = split
        self.name = None
        self.input_path = input_path
        self._wds_path = None
        
    @property
    def wds_path(self):
        return self._wds_path
    
    @wds_path.setter
    def wds_path(self, value):
        self._wds_path = value


    def download(self,
                input_format: str,
                url_col: str = "s3_uri",
                annotation_col: str = "yolo_annotations",
                output_folder: str = None,
                output_format="webdataset",
                overwrite=False) -> None:
        """
        Downloads the dataset from s3 and saves it to disk. 
        Note that we can also save directly to s3 if the output_folder is an s3 path.
        By default, the dataset is downloaded in webdataset format but we can also save it other formats supported by img2dataset.
        """
        if not output_folder:
            output_folder = self._create_output_folder("dataset")

        create_wds = self._handle_existing_output_folder(output_folder, overwrite)

        if create_wds:
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
                url_col=url_col,
                caption_col=annotation_col,
            )

        self._wds_path = output_folder

    def format_yolo(self,
            output_folder:str = None,
            wds_path:str = None,
            overwrite: bool = False) -> None:
        """
        Formats the dataset into a standard format used by yolo models.
        """
        if wds_path:
            self._wds_path = wds_path

        if not self._wds_path:
            raise ValueError("WebDataset path not specified. Please specify the path to the dataset.")

        if not output_folder:
            output_folder = self._create_output_folder("yolo")

        create_yolo = self._handle_existing_output_folder(output_folder, overwrite)

        if create_yolo:
            # get the list of tar files
            tarball_files = glob.glob(os.path.join(self._wds_path, "*.tar"))

            for tarball_file in tarball_files:
                print(f"Processing {tarball_file}...")
                self._process_tar(tarball_file, output_folder)
                print(f"Finished processing {tarball_file}.")
        print(f"Finished formatting dataset into yolo format. Dataset saved in {output_folder}.")


    @staticmethod
    def _dataset_exists(path:str):
        """
        Check if the dataset (files) exist in a local path. If it does, return True, else return False.
        """
        return os.path.exists(path) and os.listdir(path)


    def _create_output_folder(self, dataset_type: str):

        home_path = os.path.expanduser("~")
        if dataset_type == "dataset":
            output_folder = os.path.join(home_path, f"dev/data/{self.name}", dataset_type , self.split)
        elif dataset_type == "yolo":
            output_folder = os.path.join(home_path, f"dev/data/{self.name}", dataset_type)
        else:
            raise ValueError("dataset_type must be one of dataset or yolo")
        os.makedirs(output_folder, exist_ok=True)

        return output_folder

    def _handle_existing_output_folder(self, output_folder: str, overwrite: bool):

        if self._dataset_exists(output_folder):
            action = "Overwriting" if overwrite else "Skipping download"
            print(f"Dataset already exists in {output_folder}. {action}.")
            if overwrite:
                shutil.rmtree(output_folder)
            else:
                return False
        return True

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
        destination_path = os.path.join(output_folder, file_type, self.split, filename)

        # make sure the destination folder exists
        if not os.path.exists(os.path.dirname(destination_path)):
            os.makedirs(os.path.dirname(destination_path))

        with tar.extractfile(member) as src_file:
            with open(destination_path, "wb") as dst_file:
                shutil.copyfileobj(src_file, dst_file)


class TotalFusionDataset(ImageDataset):

    def __init__(self, name: str, input_path: str) -> None:
        super().__init__(name, input_path)
        self.name = "totalfusion"


class LogoDet3KDataset(ImageDataset):

    def __init__(self, split: str, input_path: str) -> None:
        super().__init__(split, input_path)
        self.name = "logodet3k"


if  __name__ == "__main__":

    tfusion_val = TotalFusionDataset(
        name="val",
        input_path="s3://mls.us-east-1.innovation/pdacosta/data/total_fusion_02/annotations/parquet/val/",
        )
    tfusion_val.download(input_format="parquet",
                        overwrite=True)
    # tfusion_val.format()

    # tfusion_train = TotalFusionDataset(
    #     name="train",
    #     input_path="s3://mls.us-east-1.innovation/pdacosta/data/total_fusion_02/annotations/parquet/train/",
    #     )
    # tfusion_train.download(input_format="parquet",
    #                     overwrite=False)

    # tfusion_train.format()


    # logodet3k_val = LogoDet3KDataset(
    #     split="val",
    #     input_path="s3://mls.us-east-1.innovation/pdacosta/data/logodet_3k/annotations/parquet/val/"
    # )
    # logodet3k_val.download(input_format="parquet",
    #                        url_col="uri",
    #                        overwrite=False)
    # logodet3k_val.format_yolo()
    
    # home_path = os.path.expanduser("~")
    # dataset_path = os.path.join(home_path, "dev/data/totalfusion/dataset", "val", "*.tar")
    # import glob
    # tarball_files = glob.glob(dataset_path)
    # webdata = wds.WebDataset(tarball_files).decode("rgb").to_tuple("jpg", "txt")
    # # webdata = webdata.batched(32)
    # i=0
    # for img, txt in webdata:
    #     if i == 19:
    #         break
    #     i+=1

    # import matplotlib.pyplot as plt
    # #save the image to disk
    # plot_boxes(img, yolo_annotation=txt, save=True)

    # plt.imsave("test.jpg", img)
    # # print(txt)
