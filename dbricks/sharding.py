from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number, lit, concat
from pyspark.sql.window import Window
import tarfile
import os
import tempfile
from math import ceil
import shutil

# Create a Spark session (Databricks automatically creates a SparkSession named "spark")
spark = SparkSession.builder \
    .appName("ImageSharding") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .getOrCreate()

# Set more parallelism for both input and output stages
spark.conf.set("spark.sql.shuffle.partitions", "1000")


# Load the input DataFrame containing image URIs from the mounted S3 bucket
input_path = "/mnt/innovation/pdacosta/data/total_fusion_02/merged/train_dataset/undersampled/"
df = spark.read.parquet(input_path)

#sample the dataframe to test the code
df = df.sample(False, 0.05, seed=0)

# Define constants and calculate the number of shards
images_per_shard = 10000
total_images = df.count()
num_shards = ceil(total_images / images_per_shard)
num_shards = int(num_shards)

print(f"Total number of shards: {num_shards}")

# Add a unique shard ID to each row in the DataFrame
window_spec = Window.orderBy(col("uri"))
df_with_shard = df.withColumn("shard_id", ((row_number().over(window_spec) - 1) / images_per_shard).cast("int"))

df_with_shard = df_with_shard.withColumn("uri", concat(lit("/mnt/"), col("uri")))




# Mounted output path
output_path = "/mnt/innovation/pdacosta/data/total_fusion_02/tar"

# Function to create and upload an archive for a shard
def create_and_upload_archive(shard_id, image_uris):
    print(f"Processing shard {shard_id} with {len(image_uris)} images")
    
    temp_dir = tempfile.mkdtemp()
    local_archive_path = os.path.join(temp_dir, f"archive_{shard_id}.tar.gz")

    # Download and add images to the local archive
    with tarfile.open(local_archive_path, "w:gz") as archive:
        for image_uri in image_uris:
            image_filename = os.path.basename(image_uri)
            local_image_path = os.path.join(temp_dir, image_filename)

            dbfs_image_uri = f"/dbfs{image_uri}"
            # Download the image directly
            shutil.copy(dbfs_image_uri, local_image_path)

            archive.add(local_image_path, arcname=image_filename)

    # Upload the archive to the mounted output path
    archive_dbfs_path = f"dbfs{output_path}/shard_{shard_id}.tar.gz"
    shutil.move(local_archive_path, archive_dbfs_path)

    # Clean up
    shutil.rmtree(temp_dir)
    print(f"Finished processing shard {shard_id}")


def process_partition(partition):
    for shard_id, image_uris in partition:
        create_and_upload_archive(shard_id, image_uris)

# Cache the DataFrame for reuse
df_with_shard.cache()

# Generate shard IDs and distribute work among partitions
shard_ids = range(num_shards)
shard_id_rdd = spark.sparkContext.parallelize(shard_ids, num_shards)

# Repartition the DataFrame to improve distribution
df_with_shard = df_with_shard.repartition(num_shards, "shard_id")


# Save each shard's archive in parallel
df_with_shard.rdd \
    .map(lambda row: (int(row["shard_id"]), row["uri"])) \
    .groupByKey() \
    .foreachPartition(lambda partition: process_partition(partition))

# # The Spark session in Databricks doesn't need explicit stopping
