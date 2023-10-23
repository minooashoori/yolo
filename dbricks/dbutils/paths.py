from pyspark.sql.functions import col, concat, lit


def replace_s3_bucket_with_dbfs(s3_path: str, is_pyspark=False):
    s3_bucket_name = s3_path.split('/')[0]
    dbfs_bucket_name = get_dbfs_mnt(s3_bucket_name, is_pyspark=is_pyspark)
    return s3_path.replace(s3_bucket_name, dbfs_bucket_name)

def get_dbfs_mnt(bucket_name: str, is_pyspark: bool):
    assert bucket_name in ['mls.godfather', 'mls.us-east-1.innovation'], "bucket name must be one of ['mls.godfather', 'mls.us-east-1.innovation']"
    if bucket_name == 'mls.godfather':
        return '/dbfs/mnt/mls.godfather' if not is_pyspark else '/mnt/mls.godfather'
    elif bucket_name == 'mls.us-east-1.innovation':
        return '/dbfs/mnt/innovation' if not is_pyspark else '/mnt/innovation'

def fix_dbfs_s3_path_for_spark(dbfs_s3_path: str):
    
    if "s3" in dbfs_s3_path:
        dbfs_s3_path = replace_s3_bucket_with_dbfs(dbfs_s3_path, is_pyspark=True)
    elif "dbfs" in dbfs_s3_path:
        dbfs_s3_path = dbfs_s3_path.replace('/dbfs', '')
    else:
        raise ValueError("dbfs_s3_path must contain either 's3' or 'dbfs'")
    
    return dbfs_s3_path

def create_uris(df):
    df = df.withColumn("s3_uri", concat(lit("s3://"), col("uri")))
    df = df.withColumn("mnt_uri", concat(lit("/mnt/"), col("uri")))
    df = df.withColumn("dbfs_uri", concat(lit("/dbfs/mnt/"), col("uri")))
    return df


def s3_to_dbfs(path):
    path.replace("s3://", "/dbfs/mnt/")
    return path

def s3_to_mnt(path):
    path.replace("s3://", "/mnt/")
    return path

def dbfs_to_s3(path):
    path.replace("/dbfs/mnt/", "s3://")
    return path

def mnt_to_s3(path):
    path.replace("/mnt/", "s3://")
    return path

def dbfs_to_mnt(path):
    path.replace("/dbfs/mnt/", "/mnt/")
    return path

def mnt_to_dbfs(path):
    path.replace("/mnt/", "/dbfs/mnt/")
    return path

def innovation_path(type="mnt"):
    if type == "mnt":
        return "/mnt/innovation"
    elif type == "dbfs":
        return "/dbfs/mnt/innovation"
    elif type == "s3":
        return "s3://mls.us-east-1.innovation"
    else:
        raise ValueError("type must be one of ['mnt', 'dbfs', 's3']")
