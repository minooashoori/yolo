from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, concat

def add_uris(df):
    df = df.withColumn("s3_uri", concat(lit("s3://"), col("uri")))
    df = df.withColumn("mnt_uri", concat(lit("/mnt/"), col("uri")))
    df = df.withColumn("dbfs_uri", concat(lit("/dbfs/mnt/"), col("uri")))
    return df

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("AddURIs") \
        .getOrCreate()

    input_path = "/mnt/innovation/pdacosta/data/total_fusion_02/merged/train_dataset/undersampled/"

    try:
        df = spark.read.parquet(input_path)
        df = add_uris(df)
        df.write.mode("overwrite").parquet(input_path)
    except Exception as e:
        print("An error occurred: ", str(e))
