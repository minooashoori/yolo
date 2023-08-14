import subprocess
import sys
import os
sys.path.append("/Workspace/Users/pdacosta@integralads.com/.ide/ctx-inference_stack/src")
from compute_engine.units.generator import UnitGenerator, UnitDeclatation
from compute_engine.structures.entities import RunInfo
from compute_engine.structures.messages import MessageHeader, RunInfoMessage, EndOfComputeMessage
from compute_engine.structures.entities import FrameDetection, FrameDetections
from compute_engine.units.handler import Handler, RunVariable
from compute_engine.utils.message_capture import MessageCapture
import pyarrow.parquet as pq
import asyncio
import pyarrow as pa
import pandas as pd
from pyspark.sql.functions import concat, lit, col

os.environ["AWS_PROFILE"]="saml"
os.environ["AWS_DEFAULT_REGION"]="eu-west-1"

n_workers = 2
model_batch_size = 16
detection_min_size_percentage = 0.01
confidence_threshold = 0.65


batch_size = 32
generate_uris = True
ignore_progress = True
dbfs_mnt_path = '/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/internal_face_detector'
pyspark_mnt_path = dbfs_mnt_path.replace('/dbfs', '')

progress_file = os.path.join(dbfs_mnt_path, 'progress.txt')


i = 0
def gen_message(url):
    global i
    run_info = {}
    run_info["export"] = 'json'
    run_info["pipeline"] = {"mode": "semi_auto"}
    run_info["source"] = {
        "kind": "image",
        "url": url,
        "uuid": "0"
    }
    run_info["run"] = {
        "id": i
    }
    i += 1
    run_info["company"] = {"id": 0}
    run_info = RunInfo(**run_info, atomic=None)

def save_batch(assets, uris, boxes, idx):
    
    
    data = [{"asset_id":asset_id, "uri": uri, "boxes": boxes} for asset_id, uri, boxes in zip(assets, uris, boxes)]
    
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    parquet_filename =f"/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/internal_face_detector/annotations/boxes_{str(idx).zfill(6)}.parquet"
    try:
        pq.write_table(table, parquet_filename)
        with open(progress_file, 'w') as f:
            f.write(str(idx))
    except:
        raise Exception('Could not save the parquet file')
    

if __name__ == '__main__':
    unit_generator = UnitGenerator(model_engine="tf")
    unit_generator.set_config("database_client", None)
    unit_generator.set_config("bucket_name", "reminiz.production")
    unit_generator.set_config("cloud_provider", "aws")
    unit_generator.set_config("models_local_path", "./")
    unit_generator.set_config("s3_bucket_url", "")


    units = []
    with unit_generator:
        input_unit = unit_generator.units.downloader()
        x = input_unit
        units.append(x)
        x @= unit_generator.units.run_frame_extractor()
        units.append(x)
        x @= unit_generator.units.frame_resizer(target_size=416)
        units.append(x)
        x @= unit_generator.units.detector(
            max_workers= n_workers,
            model_path= "models/detectors/faces/face_detector_march_20-20230116-tf/",
            batchsize= model_batch_size,
            detection_min_size_percentage= detection_min_size_percentage,
            confidence_threshold=confidence_threshold
        )


    if generate_uris:
        # get the uris from for the images
        df = spark.read.parquet("/mnt/innovation/pdacosta/data/total_fusion_02/merged/train_dataset/complete/").select("asset_id", "uri", "width", "height").distinct()

        # filter out the images that are too small, cannot have a dimension smaller than 96 px
        df = df.filter((col("width") >= 96) & (col("height") >= 96))
        
        # add the s3:// prefix
        df = df.withColumn("uri", concat(lit("s3://"), col("uri"))).select("asset_id", "uri")
        
        #save df as a csv file 
        
        df.write.mode("overwrite").csv(os.path.join(pyspark_mnt_path, "uris.csv"), header=True)
        # collect the uris and assets into a python lists
            
    else:
        # read the uris and asset_ids from the csv file
        df = spark.read.csv(os.path.join(pyspark_mnt_path, "uris.csv"), header=True)
    

    # collect the uris and assets into a python lists
    rows = df.collect()
    assets, uris = [row.asset_id for row in rows], [row.uri for row in rows]
    df.unpersist()



    if not ignore_progress:
        # read the progress index from a file
        try:
            with open(progress_file, 'r') as f:
                progress = int(f.read())
        except:
            progress = 0
    else:
        progress = 0

    n_uris = len(uris)
    print(f'Progress: {progress} out of {n_uris} images')
    
    
    # loop through the uris and call the model
    save_uris = []
    save_boxes = []
    save_assets = []
    for i in range(progress, n_uris, batch_size):
        batch_uris = uris[i:i+batch_size]
        batch_assets = assets[i:i+batch_size]
        # we call the model with the batch of uris
        # calls = [asyncio.wrap_future(input_unit(run_info=gen_message(url=uri))) for uri in uris]
        # results = await asyncio.gather(*calls)
        results = [input_unit(run_info=gen_message(url=uri)) for uri in batch_uris]
    
        batch_boxes = []
        for res in results:
            frame_detections = res[0]["frame_detections"]
            detections = frame_detections.detections
            boxes = []
            if detections:
                for frame_detection in detections:
                    box = frame_detection.box
                    boxes.append(box)
            batch_boxes.append(boxes)
        
        
        save_uris += batch_uris
        save_boxes += batch_boxes
        save_assets += batch_assets
        
        
        if (i + batch_size) % 20000 == 0:
            
            save_batch(save_assets, save_uris, save_boxes, i+batch_size)

            # reset the batch count, save uris, and save boxes
            save_uris = []
            save_boxes = []
            save_assets = []
            
    # Handle the final batch
    if save_uris:
        save_batch(save_assets, save_uris, save_boxes, n_uris)