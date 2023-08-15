import time
import pickle
import yaml
import os
import fnmatch
from typing import Union, List
import time
import json
from pyspark.sql import SparkSession
from pyspark.sql import Row
from preprocess.boxes import transform_box, intersection_area_yolo
from pyspark.sql.functions import udf
from pyspark.sql import Row
import pyspark.sql.functions as F


class ReadProcessBoxes:
    """
    This class reads pickled metadata and images from s3 and return a preprocessed merged spark dataframe
    """
    accepted_formats = ["pkl", "pickle", "parquet"]
    
    def __init__(self,
                 filepath_metadata: str = None,
                 filepath_images: str = None,
                 filepath_combined: str = None,
                 filepath_categories_map: str = None,
                 input_format: str = "pkl",
                 box_type: str = "yolo",
                 is_relative: bool = True,
                 save_path: str = None,
                 output_format: Union[str, List[str]] = "parquet",
                 output_filename: str = "metadata",
                 include_datetime: bool = False,
                 output_lib: str = "spark",
                 num_partitions: int = 48             
                 ) -> None:
        
        self.filepath_metadata = filepath_metadata
        self.filepath_images = filepath_images
        
        self.filepath_combined = filepath_combined
        
        if filepath_combined is None:
            assert filepath_metadata is not None, "filepath_metadata must be provided if filepath_combined is not provided"
            assert filepath_images is not None, "filepath_images must be provided if filepath_combined is not provided"
        else:
            assert filepath_metadata is None, "filepath_metadata must not be provided if filepath_combined is provided"
            assert filepath_images is None, "filepath_images must not be provided if filepath_combined is provided"
        
        self.filepath_categories_map = filepath_categories_map
        self.input_format = input_format

        if self.input_format not in self.accepted_formats:
            raise ValueError(f"input format not in the acceptable formats {self.acceptable_formats}")
            
        
        self.box_type = box_type
        self.is_relative = is_relative
        assert save_path is not None, "save_path must be provided"
        self.save_path = save_path
        self.output_format = output_format
        self.num_partitions = num_partitions
        
        if include_datetime:
            # include YYYYMMDD_HHMMSS in the output filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_filename = f"{output_filename}_{timestamp}"
        else:
            self.output_filename = output_filename
        
        self.include_datetime = include_datetime
        self.output_lib = output_lib
        
        
        # check if the file exists and it's a YAML file
        if filepath_categories_map is not None:
            self.categories_map = self._load_categories_map()
        else:
            self.categories_map = None
            

    
        
    def _load_categories_map(self):
        """
        Load the categories map from the YAML file
        """
        # check if the file exists and it's a YAML file
        assert os.path.exists(self.filepath_categories_map), f"Category map file: {self.filepath_categories_map} does not exist"
        
        with open(self.filepath_categories_map, 'r') as yaml_file:
            categories_map = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return categories_map
    
    
    def _list_and_match_files(self, path: str, file_pattern: str):
        
        # list files in dbfs
        # if path is a file, then return the file
        # if not then return all files matching the pattern
        if os.path.isfile(path) and path.endswith(".pkl") or path.endswith(".pickle"):
            # check if the file extension matches the allowed extensions
                return [path]            
        
        files = os.listdir(path)    
        matching_files = fnmatch.filter(files, file_pattern)
        # add the path to the matching files
        matching_files = [os.path.join(path, file) for file in matching_files]
        
        
        if not matching_files:
            raise FileNotFoundError(f"No files found matching {file_pattern}")
        
        return matching_files
    
    def _read_pickle(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    
    def _load_data(self, filepath, spark: SparkSession = None, return_as: str = "rdd"):
        """
        Load the dataframe from pickle file and return it as an RDD
        """
        
        if return_as not in ["rdd", "list"]:
            raise ValueError(f"return_as must be one of ['rdd', 'list'] but got {return_as}")
        
        if spark is None:
                raise ValueError("spark must be provided if return_as='rdd'")
        
        if self.input_format in ["pkl", "pickle"]:
            try:
                matching_files = self._list_and_match_files(filepath, file_pattern="*.pkl")
            except FileNotFoundError:
                matching_files = self._list_and_match_files(filepath, file_pattern="*.pickle")
            
            data = {}
            for file in matching_files:
                data.update(self._read_pickle(file))
        
            valid_data = {key: value for key, value in data.items() if value is not None and key != "last_idx"}
            
            if return_as in ["list"]:  
                return list(valid_data.items())
            elif return_as in ["rdd"]:

                return spark.sparkContext.parallelize(valid_data.items(), self.num_partitions)      
        
        if self.input_format in ["parquet"]:
            
            # check if /dbfs/ is in the path if so, we have to remove it to use in the spark.read.parquet
            if filepath.startswith("/dbfs"):
                filepath = filepath.replace("/dbfs", "")
            
            # read the parquet files from the path
            valid_data = spark.read.parquet(filepath)

            if return_as in ["list"]:
                raise ValueError("return_as='list' is not supported for parquet files")
            
            elif return_as in ["rdd"]:
                return valid_data.rdd.repartition(self.num_partitions)
    
    
    def _transform_box_to_row(self, box, box_type, category_id):
        
        box = [float(coord) for coord in box] # make sure all coordinates are floats
        
        if box_type in ["yolo", "xywh"]:
            x, y, w, h = box
            transf_row_box = Row(category=category_id, x=x, y=y, width=w, height=h)     
        else:
            x1, y1, x2, y2 = box
            transf_row_box = Row(category=category_id, x1=x1, y1=y1, x2=x2, y2=y2)
            
        return transf_row_box
    
            
    
    def _get_category_id(self, category, categories_map):
        
        if not categories_map:
            return category
        else:
            if category in categories_map:
                return categories_map[category]["id"]
            else:
                return None


    def _preprocess_metadata_rdd(self, row, box_type, is_relative, categories_map=None, is_combined=False):
    
        if self.input_format in ["pkl", "pickle"]:
            asset_id, metadata = row
            image_size = metadata['image_size']
            boxes = metadata['boxes']
            if is_combined:
                uri = metadata["uri"]

            if isinstance(image_size, str):
                # there is an issue we have to fix, this if statement is a temporary fix
                image_size = boxes['image_size']
                boxes = boxes['boxes']
            
        else:
            asset_id = row.asset_id
            image_size = row.image_size
            boxes = row.boxes
            if is_combined:
                uri = row.uri
            
        width, height = image_size

        # Restructure "boxes" list of dictionaries
        transformed_boxes = []
        
        for box in boxes:
            
            box_coords = box['box'] # box coord is a mix of absolute and relative xyxy coordinates
            category = box['category']
               
            category_id = self._get_category_id(category, categories_map)
            
            if category_id is None:
                continue        
 
            transf_box = transform_box(box_coords, image_size, box_type, is_relative)

            transf_row_box = self._transform_box_to_row(transf_box, box_type, category_id)

            transformed_boxes.append(transf_row_box)
                
        # Return only the transformed metadata as a single Row
        if not is_combined:
            return Row(
                asset_id=asset_id,
                width=width,
                height=height,
                boxes=transformed_boxes,
                box_type=box_type 
            )
        else:
            
            return Row(
                asset_id=asset_id,
                width=width,
                height=height,
                boxes=transformed_boxes,
                uri=uri,
                box_type=box_type 
            )
                
    def _merge_dataframes(self, df_metadata, df_images):
        
        return df_metadata.join(df_images, on="asset_id", how="inner")
    
    def _save_dataframe(self, df, path_to_file, format):
        
       base_path = os.path.dirname(path_to_file)
       if not os.path.exists(base_path):
              os.makedirs(base_path) 
        
       if format=="parquet":
           #make sure that if the path includes dbfs, it is removed
           path_to_file = path_to_file.replace("/dbfs", "")
           
           df.write.mode("overwrite").parquet(path_to_file+"/")

       elif format in ["pickle", "json"]:
            df_list = df.collect() # this is an expensive operation depending on the size of the dataframe

            df_map = {row.asset_id: {
                "width": row.width,
                "height": row.height,
                "boxes": [box.asDict() for box in row.boxes] if row.boxes else [],
                "uri": row.uri,
                "box_type": row.box_type
            } for row in df_list}
            
                
            if format == "pickle":
                with open(path_to_file+".pkl", "wb") as file:
                    pickle.dump(df_map, file)
            elif format == "json":
                with open(path_to_file+".json", "w") as file:
                    json.dump(df_map, file)
                    
             
    def read(self):
        
        is_combined = True if self.filepath_combined else False
        
        spark = SparkSession.builder.appName("Reader").getOrCreate()
        if not is_combined:
            print("Reading metadata...")
            meta_rdd = self._load_data(self.filepath_metadata, spark=spark)
            print("Reading images...")
            img_rdd = self._load_data(self.filepath_images, spark=spark)
            print("Successfully read metadata and image uris")
        else:
            meta_rdd = self._load_data(self.filepath_combined, spark=spark)
        
        print("Checking if a categories map was provided...")
        if not self.categories_map:
            print("No categories map provided. Proceeding without it...")
        else:
            print(f"Categories map provided from {self.filepath_categories_map}. Proceeding with it")
            
        # show which categories are considered in the categories map
        print(f"Categories considered in the categories map: {[self.categories_map[key]['description'] for key in self.categories_map.keys()]}")
        
        
        print("Processing metadata...")
        processed_rdd = meta_rdd.map(lambda row: self._preprocess_metadata_rdd(row, self.box_type, self.is_relative, self.categories_map, is_combined))
        df = spark.createDataFrame(processed_rdd)

        if self.filepath_combined is None:
            print("Processing images...")
            img_df = spark.createDataFrame(img_rdd, ["asset_id", "uri"])
            
            print("Merging metadata and images...")
            df = self._merge_dataframes(df, img_df)
            

        print(f"Final dataframe contains {df.count()} rows")
        
        # print the top row of the dataframe
        print("Top row: ")
        print(df.head())
        # print number of boxes of the top row
        print(f"Number of boxes in the top row: {len(df.head().boxes)}")
        
        # count the number of rows with an empty boxes list
        print(f"Number of rows with empty boxes list: {df.filter(F.size(F.col('boxes')) == 0).count()}")
        
        # print the schema of the dataframe
        print("Schema: ")
        print(df.printSchema())        
        
        if isinstance(self.output_format, str):
            self.output_format = [self.output_format]
        
        if "parquet" not in self.output_format:
            self.output_format = ["parquet"] + self.output_format
            
        
        for format in self.output_format:
            print(f"Saving dataframe to {self.save_path} in {format} format with filename {self.output_filename}")
            self._save_dataframe(df, self.save_path+self.output_filename, format)

        

if __name__ == "__main__":
    os.environ["AWS_PROFILE"]="saml"
    os.environ["AWS_DEFAULT_REGION"]="eu-west-1"
    
    # print("Reading train dataset...")
    # reader_train = ReadProcessBoxes(
    #     filepath_metadata="/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/train_dataset/metadata.pkl",
    #     filepath_images="/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/train_dataset/asset_paths.pkl",
    #     filepath_categories_map="/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/map/config.yaml",   
    #     save_path="/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/merged/train_dataset/",        
    # )
    
    # reader_train.read()
    
    # reader_train_part_2 = ReadProcessBoxes(
    #     filepath_combined="/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/train_dataset/extra/output/",
    #     filepath_categories_map="/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/map/config.yaml",
    #     input_format="parquet",
    #     save_path="/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/merged/train_dataset/extra/",
    # )
    # reader_train_part_2.read()
    
    # stack the two parts of the train dataset and save it
    print("Stacking the two parts of the train dataset...")
    train_df_1 = spark.read.parquet("/mnt/innovation/pdacosta/data/total_fusion_02/merged/train_dataset/metadata/")
    train_df_2 = spark.read.parquet("/mnt/innovation/pdacosta/data/total_fusion_02/merged/train_dataset/extra/metadata/")
    
    
    train_df_1 = train_df_1.select("asset_id", "width", "height", "boxes", "uri", "box_type")
    train_df_2 = train_df_2.select("asset_id", "width", "height", "boxes", "uri", "box_type")
    
    # stack the two dataframes and remove duplicates based on asset_id
    train_df = train_df_1.union(train_df_2).dropDuplicates(["asset_id"])

    
    # save the stacked dataframe
    train_df.write.mode("overwrite").parquet("/mnt/innovation/pdacosta/data/total_fusion_02/merged/train_dataset/complete/")
    
        
    # print("Reading test dataset...")
    # reader_test = ReadProcessBoxes(
    #     filepath_combined="/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/test_dataset/",
    #     filepath_categories_map="/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/map/config.yaml",
    #     save_path="/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/merged/test_dataset/",
    #     output_format=["pickle"]
    # )

    # reader_test.read()
        
           
            
    
    
        
        
            
            
    