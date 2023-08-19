import os
from typing import List, Union
from pyspark.sql import SparkSession
from distributor import _spark_session
from preprocess.paths import replace_s3_bucket_with_dbfs 
from preprocess.paths import fix_dbfs_s3_path_for_spark
import shutil
import math
import yaml
import tarfile

class YoloWriter:
    
    def __init__(self,
                 metadata_filepath: str,
                 output_folder: str,
                 split: bool,
                 tar_folder: str = "/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/yolo",
                 category_map_filepath: str = None,
                 override_category_map: dict = None,
                 splits: Union[List[str], str] = ["train", "val"],
                 split_weights: List[float] = [0.8, 0.2],
                 sample_fraction: float = 0.5,
                 seed: int = 42,
                 n_processes: int = 32,
                 images_only: bool = False
                 ) -> None:
        
        # path block
        self.metadata_filepath = metadata_filepath
        self.output_folder = output_folder
        self.dataset_name = self.output_folder.split('/')[-1]
        self.output_path_images = os.path.join(output_folder, "images")
        self.output_path_labels = os.path.join(output_folder, "labels")
        self.category_map_filepath = category_map_filepath
        if override_category_map:
            self.category_map = override_category_map
        else:
            self.category_map = self._load_categories_map()
        self.n_classes = len(self.category_map)
        
        self.tar_folder = tar_folder
        
        # split block
        self.split = split
        self.splits = splits
        self.split_weights = split_weights
        self._check_splits()

        # sampling block
        self.sample_fraction = sample_fraction
        assert 0.0 < self.sample_fraction <= 1.0, "sample_fraction must be between 0.0 and 1.0"
            
        # other
        self.seed = seed
        self.n_processes = n_processes
        self.images_only = images_only
        
    
    
    def _check_splits(self):
        valid_splits = ["train", "val", "test"]
        
        if self.split:
            print("Variable split is True, checking splits and split_weights...")
            # Ensure splits is a list and only contains valid splits
            self.splits = [self.splits] if isinstance(self.splits, str) else self.splits
            assert all(split in valid_splits for split in self.splits), f"Invalid split(s): {', '.join(set(self.splits) - set(valid_splits))}. Valid splits are: {', '.join(valid_splits)}"
            
            # Ensure splits has the same length as split_weights
            assert len(self.splits) == len(self.split_weights), "splits and split_weights must have the same length"
            
            # Warn about normalization of split_weights if their sum is not close to 1.0
            if not math.isclose(sum(self.split_weights), 1.0):
                print("Warning: the sum of the split_weights is not 1.0. The split_weights will be normalized")
            
            # reorder splits and split_weights according to the order of valid_splits
            self.splits = [split for split in valid_splits if split in self.splits]
            self.split_weights = [self.split_weights[self.splits.index(split)] for split in valid_splits if split in self.splits] 
            
            print("All good, moving on.")
        else:
            # If split is False, set default splits and split_weights
            self.splits = ["train"]
            self.split_weights = [1.0]
            print("Warning: splits and split_weights will be ignored")


    
    def _load_categories_map(self):
        """
        Load the categories map from the YAML file
        """
        # check if the file exists and it's a YAML file
        assert os.path.exists(self.category_map_filepath), f"Category map file: {self.category_map_filepath} does not exist"
        
        with open(self.category_map_filepath, 'r') as yaml_file:
            categories_map = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return categories_map
    
    def _create_folder_structure(self):
        # make sure the output folder exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        # same for images and labels
        if not os.path.exists(self.output_path_images):
            os.makedirs(self.output_path_images)
        if not os.path.exists(self.output_path_labels):
            os.makedirs(self.output_path_labels)
            
        
    def _read_data(self, spark: SparkSession):
        
        return spark.read.parquet(fix_dbfs_s3_path_for_spark(self.metadata_filepath))
    
    def _sample_data(self, df):
        if self.sample_fraction == 1.0:
            return df
        return df.sample(withReplacement=False, fraction=self.sample_fraction, seed=self.seed)
    
    def _split_data(self, df):
        # split the data into train and valid
        if len(self.splits) == 1:
            return df
        elif len(self.splits) == 2:
            train_df, valid_df = df.randomSplit(self.split_weights)
            return train_df, valid_df
        elif len(self.splits) == 3:
            train_df, valid_df, test_df = df.randomSplit(self.split_weights)
            return train_df, valid_df, test_df
        
    def _write_image(self, uri: str, image_filename: str,  split: str):
    
        # replace the bucket name with the dbfs mount name 
        source_path = replace_s3_bucket_with_dbfs(uri)
        # create the destination path for the image
        destination_path = os.path.join(self.output_path_images, split, image_filename)
        
        # copy the image
        shutil.copyfile(source_path, destination_path)
    
    
    def _write_row(self, row, split: str):
        uri = row.uri
        boxes = row.boxes
        
        # get the image filename from the uri
        image_filename = uri.split('/')[-1]
        
        if not self.images_only:
            self._write_annotations(boxes, image_filename, split)
        
        self._write_image(uri, image_filename, split)
    
    def _process_partition(self, rows, split: str):
    
    
        for row in rows:
            uri = row.uri
            boxes = row.boxes
            image_filename = uri.split('/')[-1]

            if not self.images_only:
                self._write_annotations(boxes, image_filename, split)

            self._write_image(uri, image_filename, split)

        
    
    def _write_annotations(self, boxes, image_filename: str, split: str):
        
        # remove only the last extension e.g img.jpg.jpg -> img.jpg.txt
        image_name = os.path.splitext(image_filename)[0]
        
        # if we have boxes
        if len(boxes) != 0:
        
            content = ""
            for box in boxes:
                category = box.category
                x, y, w, h = box.x, box.y, box.width, box.height
                line = f"{category} {x} {y} {w} {h}\n"
                content += line
            
            # create the annotation file
            annotation_filename = f"{image_name}.txt"
            annotation_filepath = os.path.join(self.output_path_labels, split, annotation_filename)
            with open(annotation_filepath, "w") as f:
                f.write(content)
            
    def _write_dataset_yaml(self):
        
        data_yaml = ""
        
        for split in self.splits:
            data_yaml += f"{split}: {self.output_path_images}/{split}\n"
        # add the number of classes
        # add a line
        data_yaml += "\n"
        # add a comment
        data_yaml += "# number of classes\n"
        data_yaml += f"nc: {self.n_classes}\n"
        
        # add a line
        data_yaml += "\n"
        # add a comment
        # add the class names
        id_dict = {v["id"]: v["description"] for _, v in self.category_map.items()}
        # sort the dictionary by key
        id_dict = dict(sorted(id_dict.items()))
        data_yaml += "# class names\n"
        data_yaml += f"names: {list(id_dict.values())}"
        
        # we are done, write the file
        with open(os.path.join(self.output_folder, "dataset.yaml"), "w") as f:
            f.write(data_yaml)
    
    def _create_tar_archive(self):
        # Create a tar archive of the entire output folder
        tar_filename = os.path.join(self.tar_folder, self.dataset_name, "data.tar")
        with tarfile.open(tar_filename, "w") as tar:
            tar.add(self.output_folder, arcname="yolo")
            


        
    def write(self):
        # the final folder structure should be like this:
        # - yolo_dataset
        #     - images
        #         - train
        #             - image1.jpg
        #         - valid
        #             - image2.jpg
        #         - test
        #             - image3.jpg
        #     - labels
        #         - train
        #             - image1.txt
        #         - valid
        #             - image2.txt
        #         - test
        #             - image3.txt
        #     - dataset.yaml
        
        self._create_folder_structure()
        
        # connect to spark
        with _spark_session(processes_count=self.n_processes) as spark:
        
            # read the data (parquet)
            df = self._read_data(spark)
            
            # partition the data
            df = df.repartition(2000)
            
            # # sample if necessary
            df = self._sample_data(df)
            print(f"After sampling (or not) we have {df.count()} images.")
            
            
            # split if necessary
            if self.split:
                dfs = self._split_data(df)
            else:
                dfs = [df]
                
            
                
            for split, df in zip(self.splits, dfs):
                # create the folders if they don't exist
                os.makedirs(os.path.join(self.output_path_images, split), exist_ok=True)
                os.makedirs(os.path.join(self.output_path_labels, split), exist_ok=True)
                # for each split, write the images and annotations
                # df.foreach(lambda row: self._write_row(row, split))
                df.foreachPartition(lambda rows: self._process_partition(rows, split))
                
                # save the dataframe with the corresponding split
                df.write.mode("overwrite").parquet(os.path.join(self.tar_folder.replace("/dbfs", ""), self.dataset_name, f"{split}_asset_ids.parquet"))   
            # # write the dataset.yaml file
            self._write_dataset_yaml()
            
            print("Creating tar archive...")
            self._create_tar_archive()
            
            
            # # stop spark
            # spark.stop()
            
            print("Done!")
        
if __name__ == "__main__":
    os.environ["AWS_PROFILE"]="saml"

    yolo_writer = YoloWriter(
        metadata_filepath="/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/merged/train_dataset/undersampled/",
        output_folder="/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/yolo_60k_under",
        override_category_map={134533: {"description": "face", "id": 0}, 90020: {"description": "logo", "id": 1}},
        split=True,
        splits = ["train", "val", "test"],
        split_weights=[0.7, 0.2, 0.1],
        sample_fraction=0.1
    )
    yolo_writer.write()