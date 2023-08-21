import glob
import os
import pandas as pd

class YOLODatasetWriter:
    
    
    def __init__(self, 
                 class_map,
                 annot_train_path,
                 annot_val_path=None,
                 annot_test_path=None,
                 ext="csv",
                 dataset_path=None):
        self.class_map = class_map
        self.annot_train_path = annot_train_path
        self.annot_val_path = annot_val_path
        self.annot_test_path = annot_test_path
        self.ext = ext
        self.dataset_path = dataset_path
        
    def _read_annotaton_file(self, path):
        df = pd.read_csv(path)
        return df

    def _list_files(self, path):
        # with extension csv
        files = glob.glob(os.path.join(path, f"*.{self.ext}"))
        # check if there is only one file
        if len(files) == 1:
            return files[0]
        else:
            raise ValueError(f"More than one file found in {path}")
        
    def _create_annotations(self, df, split):
        
        # for each row we will create a txt file with name "asset".txt and the content will bee "yolo_annotations"
        for index, row in df.iterrows():
            asset = row["asset"]
            yolo_annotations = row["yolo_annotations"]
            txt_file_path = os.path.join(self.dataset_path, split, f"{asset}.txt")
            # if dir does not exist create it
            os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
            print(f"Creating {txt_file_path}")
            with open(txt_file_path, "w") as f:
                f.write(yolo_annotations)
    
    def create_dataset(self):
        
        train_file = self._list_files(self.annot_train_path)
        val_file   = self._list_files(self.annot_val_path)
        test_file  = self._list_files(self.annot_test_path)
        
        train_df = self._read_annotaton_file(train_file)
        val_df = self._read_annotaton_file(val_file)
        test_df = self._read_annotaton_file(test_file)
        
        self._create_annotations(train_df, "train")
        self._create_annotations(val_df, "val")
        self._create_annotations(test_df, "test")
        
            
base_path =  "/home/ec2-user/dev/data/logodet3k/annotations"
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")
test_path = os.path.join(base_path, "test")

writer = YOLODatasetWriter(None, train_path, val_path, test_path, ext="csv", dataset_path="/home/ec2-user/dev/data/logodet3k/yolo_dataset/labels_")
writer.create_dataset()