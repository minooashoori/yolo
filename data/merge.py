import os
import glob
import shutil
from tqdm import tqdm

class MergeImgDataset:

    """
    Merge two yolo image datasets
    """


    def __init__(self,
                datasetA_path = None,
                datasetB_path = None,
                output_folder = None) -> None:
        self.datasetA_path = datasetA_path
        self.datasetB_path = datasetB_path
        self.output_folder = output_folder
        # if self.output_folder exists, delete it
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        self.output_images_folder = self._create_output_folder(os.path.join(output_folder, "images"))
        self.output_labels_folder = self._create_output_folder(os.path.join(output_folder, "labels"))
        self._create_subfolders(self.output_images_folder, ["train", "val", "test"])
        self._create_subfolders(self.output_labels_folder, ["train", "val", "test"])

    def _list_folders(self, path):

        # list all folders in path, we assume that each folder is a dataset with  images and labels and potential subfolders train, val, test
        folders = glob.glob(os.path.join(path, "*", "*"))
        folders = [folder for folder in folders if os.path.isdir(folder)]
        return folders

    def _list_files(self, path, ext):
        # list all files in path with extension ext
        files = glob.glob(os.path.join(path, f"*.{ext}"))
        return files

    def _list_images(self, path):
        return self._list_files(path, "jpg")

    def _list_labels(self, path):
        return self._list_files(path, "txt")

    def _list_subfolders(self, path, subfolder: str):
        folders = self._list_folders(path)
        subfolders = [folder for folder in folders if folder.endswith(subfolder)]
        return subfolders

    def _copy_files(self, files, dest, prefix=""):
        for file in tqdm(files, desc=f"Copying files:"):
            # We will add a prefix to the filename in the destination folder
            filename = os.path.basename(file)
            dest_ = os.path.join(dest, f"{prefix}{filename}")
            shutil.copy(file, dest_)

    def _create_output_folder(self, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def _get_folder_type(self, folders, type_):

        for folder in folders:
            # if there's a folder with type_ but not necessarily the last folder
            if  folder.find(type_) != -1:
                return folder
        return None

    def _create_subfolders(self, output_folder: str, subfolders: list):
        for subfolder in subfolders:
            os.makedirs(os.path.join(output_folder, subfolder), exist_ok=True)


    def _copy_and_prefix(self, foldersA, foldersB, split):

        if foldersA:
            imagesA = self._list_images(self._get_folder_type(foldersA, "images"))
            labelsA = self._list_labels(self._get_folder_type(foldersA, "labels"))
            print(f"Copying {len(imagesA)} images and {len(labelsA)} labels from dataset A, split: {split}...")
            self._copy_files(imagesA, os.path.join(self.output_images_folder, split), "A_")
            self._copy_files(labelsA, os.path.join(self.output_labels_folder, split), "A_")
        if foldersB:
            imagesB = self._list_images(self._get_folder_type(foldersB, "images"))
            labelsB = self._list_labels(self._get_folder_type(foldersB, "labels"))
            print(f"Copying {len(imagesB)} images and {len(labelsB)} labels from dataset B, split: {split}...")
            self._copy_files(imagesB, os.path.join(self.output_images_folder, split), "B_")
            self._copy_files(labelsB, os.path.join(self.output_labels_folder, split), "B_")


    def merge(self, splits=list[str],
            includeA=["train", "val", "test"],
            includeB=["train", "val", "test"]):
        # get train folders in the two datasets
        train_foldersA = self._list_subfolders(self.datasetA_path, "train") if "train" in includeA else None
        train_foldersB = self._list_subfolders(self.datasetB_path, "train") if "train" in includeB else None

        # get val folders in the two datasets
        val_foldersA = self._list_subfolders(self.datasetA_path, "val") if "val" in includeA else None
        val_foldersB = self._list_subfolders(self.datasetB_path, "val") if "val" in includeB else None

        # get test folders in the two datasets
        test_foldersA = self._list_subfolders(self.datasetA_path, "test") if "test" in includeA else None
        test_foldersB = self._list_subfolders(self.datasetB_path, "test") if "test" in includeB else None


        if "train" in splits:
            # copy train images and labels
            self._copy_and_prefix(train_foldersA, train_foldersB, "train")
        if "val" in splits:
            # copy val images and labels
            self._copy_and_prefix(val_foldersA, val_foldersB, "val")
        if "test" in splits:
            # copy test images and labels
            self._copy_and_prefix(test_foldersA, test_foldersB, "test")



if __name__ == "__main__":

    merge = MergeImgDataset(
        datasetA_path="/home/ec2-user/dev/data/totalfusion/yolo",
        datasetB_path="/home/ec2-user/dev/data/logodet3k/yolo",
        output_folder="/home/ec2-user/dev/data/totalfusion_logodet")

    merge.merge(splits=["train", "val"], includeA=["train", "val"], includeB=["train"])