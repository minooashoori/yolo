import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"img2dataset"))

import img2dataset as i2d
import webdataset as wds
import tarfile
import shutil
import glob
import random
import json

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
                overwrite=False,
                resize_mode='keep_ratio_largest',
                resize_only_if_bigger=True) -> None:
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
                processes_count=8,
                thread_count=8,
                url_list=self.input_path,
                output_folder=output_folder,
                output_format=output_format,
                input_format=input_format,
                number_sample_per_shard=10000,
                distributor="multiprocessing",
                resize_mode=resize_mode,
                resize_only_if_bigger=resize_only_if_bigger,
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
            overwrite: bool = False,
            prop: float = 1.0,
            keep_original_filenames=False,
            get_original_url_width_height=False) -> None:
        """
        Formats the dataset into a standard format used by yolo models.
        """
        if wds_path:
            self._wds_path = wds_path

        if not self._wds_path:
            raise ValueError("WebDataset path not specified. Please specify the path to the dataset.")


        output_folder = output_folder or self._create_output_folder("yolo")
        images_folder = os.path.join(output_folder, "images", self.split)
        labels_folder = os.path.join(output_folder, "labels", self.split)
        
   

        create_yolo = self._handle_existing_output_folder(images_folder, overwrite) and self._handle_existing_output_folder(labels_folder, overwrite)

        urls_, widths_, heights_ = [], [], []
        if create_yolo:
            # get the list of tar files

            tarball_files = glob.glob(os.path.join(self._wds_path, "*.tar"))

            for tarball_file in tarball_files:
                print(f"Processing {tarball_file}...")
                urls, widths, heights = self._process_tar(tarball_file, output_folder, prop, keep_original_filenames, get_original_url_width_height)
                print(f"Finished processing {tarball_file}.")
                urls_.extend(urls), widths_.extend(widths), heights_.extend(heights)
        print(f"Finished formatting dataset into yolo format. Dataset saved in {output_folder}.")
        
        if get_original_url_width_height:
            return urls_, widths_, heights_


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
    
    
    
    def _get_original_url_width_height(self, tar, member):
        # json file with the same name
        # if it's a  .jpg file, we need to get the corresponding json file
        if member.name.endswith(".jpg"):
            json_filename = member.name.replace(".jpg", ".json")
        elif member.name.endswith(".txt"):
            json_filename = member.name.replace(".txt", ".json")
        # read the original url from the json file 
        dict_ = json.load(tar.extractfile(json_filename))
        url = dict_["url"]
        width = dict_["original_width"]
        height = dict_["original_height"]
        return url, width, height


    def _get_original_filename(self, tar, member):

         # json file with the same name
        if member.name.endswith(".jpg"):
            json_filename = member.name.replace(".jpg", ".json")
        elif member.name.endswith(".txt"):
            json_filename = member.name.replace(".txt", ".json")
        # read the original url from the json file 
        dict_ = json.load(tar.extractfile(json_filename))
        url = dict_["url"]
        # get the filename from the url without the extension
        filename = os.path.basename(url).split(".")[0]
        return filename

    def _process_tar(self, tar_path, output_folder, prop: float, keep_original_filenames: bool = False, get_original_url_width_height: bool =False):
        copied_images = set()
        urls, widths, heights = [], [], []
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith((".jpg", ".txt")):
                    filename = None
                    if member.name.endswith(".jpg") and random.random() < prop:
                        print(f"Processing {member.name}...")
                        if keep_original_filenames:
                            filename = self._get_original_filename(tar, member)
                        if get_original_url_width_height:
                            url, width, height = self._get_original_url_width_height(tar, member)
                            urls.append(url), widths.append(width), heights.append(height)
                        self._copy_file(tar, member, output_folder, "images", filename)
                        copied_images.add(member.name)
                    elif member.name.endswith(".txt"):
                        jpg_filename = member.name.replace(".txt", ".jpg")
                        if jpg_filename in copied_images and (member.size > 20 or self._is_non_empty_txt_file(tar, member)):
                            if keep_original_filenames:
                                filename = self._get_original_filename(tar, member)
                            self._copy_file(tar, member, output_folder, "labels", filename)
        return urls, widths, heights


    def _copy_file(self, tar, member, output_folder, file_type, filename=None):

        assert file_type in ["images", "labels"], "file_type must be one of images or labels"
        
        if not filename:
            filename = os.path.basename(member.name)
        else:
            # check if we have an extension
            if not filename.endswith((".jpg", ".txt")):
                filename = filename + ".jpg" if file_type == "images" else filename + ".txt"
        destination_path = os.path.join(output_folder, file_type, self.split, filename)
        print(f"Copying {member.name} to {destination_path}...")

        # make sure the destination folder exists
        if not os.path.exists(os.path.dirname(destination_path)):
            os.makedirs(os.path.dirname(destination_path))

        with tar.extractfile(member) as src_file:
            with open(destination_path, "wb") as dst_file:
                shutil.copyfileobj(src_file, dst_file)

    def _is_non_empty_txt_file(self, tar, member):
        with tar.extractfile(member) as src_file:
            content = src_file.read()
            return bool(content)


class TotalFusionDataset(ImageDataset):

    def __init__(self, name: str, input_path: str) -> None:
        super().__init__(name, input_path)
        self.name = "totalfusion"


class LogoDet3KDataset(ImageDataset):

    def __init__(self, split: str, input_path: str) -> None:
        super().__init__(split, input_path)
        self.name = "logodet3k"

class Logo05Dataset(ImageDataset):

    def __init__(self, split: str, input_path: str) -> None:
        super().__init__(split, input_path)
        self.name = "logo05"

class Logo05FusionDataset(ImageDataset):

    def __init__(self, split: str, input_path: str) -> None:
        super().__init__(split, input_path)
        self.name = "logo05fusion"

class PortalDataset(ImageDataset):

    def __init__(self, split: str, input_path: str) -> None:
        super().__init__(split, input_path)
        self.name = "portal"


if  __name__ == "__main__":

    # tfusion_val = TotalFusionDataset(
    #     name="val",
    #     input_path="s3://mls.us-east-1.innovation/pdacosta/data/total_fusion_02/annotations/parquet/val/",
    #     )
    # tfusion_val.download(input_format="parquet",
    #                     overwrite=True)
    # tfusion_val.format_yolo(overwrite=True,
    #     prop=0.1)

    # tfusion_train = TotalFusionDataset(
    #     name="train",
    #     input_path="s3://mls.us-east-1.innovation/pdacosta/data/total_fusion_02/annotations/parquet/train/",
    #     )
    # tfusion_train.download(input_format="parquet",
    #                     overwrite=False)

    # tfusion_train.format_yolo()


    # logodet3k_val = LogoDet3KDataset(
    #     split="val",
    #     input_path="s3://mls.us-east-1.innovation/pdacosta/data/logodet_3k/annotations/parquet/val/"
    # )
    # logodet3k_val.download(input_format="parquet",
    #                        url_col="uri",
    #                        overwrite=False)
    # logodet3k_val.format_yolo()

    # logodet3k_train = LogoDet3KDataset(
    #     split="train",
    #     input_path="s3://mls.us-east-1.innovation/pdacosta/data/logodet_3k/annotations/parquet/train/"
    # )
    # logodet3k_train.download(input_format="parquet",
    #                        url_col="uri",
    #                        overwrite=False)
    # logodet3k_train.format_yolo()
    
    # logo05_test =  Logo05Dataset(
    #     split="test",
    #     input_path="s3://mls.us-east-1.innovation/pdacosta/data/logo05/annotations/gts_preds/parquet/xywh/test/"
    # )
    # logo05_test.download(input_format="parquet",
    #                       url_col="uri",
    #                       resize_mode="no",
    #                       annotation_col=None,
    #                       overwrite=False
    # )
    # logo05_test.format_yolo(overwrite=True, keep_original_filenames=True)
    
    # logo05fusion_train = Logo05FusionDataset(
    #     split="train",
    #     input_path = "s3://mls.us-east-1.innovation/pdacosta/data/total_fusion_02/annotations/parquet/logo_05/train/",
    # )
    # logo05fusion_train.download(input_format="parquet",
    #                             url_col="s3_uri",
    #                             annotation_col="yolo_annotations",
    #                             overwrite=True)
    
    # logo05fusion_train.format_yolo(overwrite=True)
    
    # logo05fusion_val = Logo05FusionDataset(
    #     split="val",
    #     input_path = "s3://mls.us-east-1.innovation/pdacosta/data/total_fusion_02/annotations/parquet/logo_05/val/",
    # )
    # logo05fusion_val.download(input_format="parquet",
    #                             url_col="s3_uri",
    #                             annotation_col="yolo_annotations",
    #                             overwrite=True)
    
    # logo05fusion_val.format_yolo(overwrite=True)
    
    # portal_train = PortalDataset(
    #     split="train",
    #     input_path = "/home/ec2-user/dev/data/portal/train.parquet",
    # )
    
    # portal_train.download(input_format="parquet",
    #                         url_col="s3_uri",
    #                         annotation_col="yolo_annotations",
    #                         overwrite=False)
    
    # portal_train.format_yolo(overwrite=True, keep_original_filenames=True, get_original_url_width_height=False)
    
    
    portal_test = PortalDataset(
        split="val",
        input_path = "/home/ec2-user/dev/data/portal/test.parquet",
    )
    
    portal_test.download(input_format="parquet",
                            url_col="s3_uri",
                            annotation_col="yolo_annotations",
                            overwrite=True)
    
    portal_test.format_yolo(overwrite=True, keep_original_filenames=True, get_original_url_width_height=False)
    
    
