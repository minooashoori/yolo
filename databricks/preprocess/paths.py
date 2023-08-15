def replace_s3_bucket_with_dbfs(s3_path, is_pyspark=False):
    s3_bucket_name = s3_path.split('/')[0]
    dbfs_bucket_name = get_dbfs_mnt(s3_bucket_name, is_pyspark=is_pyspark)
    return s3_path.replace(s3_bucket_name, dbfs_bucket_name)

def get_dbfs_mnt(bucket_name, is_pyspark):
    assert bucket_name in ['mls.godfather', 'mls.us-east-1.innovation'], "bucket name must be one of ['mls.godfather', 'mls.us-east-1.innovation']"
    if bucket_name == 'mls.godfather':
        return '/dbfs/mnt/mls.godfather' if not is_pyspark else '/mnt/mls.godfather'
    elif bucket_name == 'mls.us-east-1.innovation':
        return '/dbfs/mnt/innovation' if not is_pyspark else '/mnt/innovation'

def fix_dbfs_s3_path_for_spark(dbfs_s3_path):
    
    if "s3" in dbfs_s3_path:
        dbfs_s3_path = replace_s3_bucket_with_dbfs(dbfs_s3_path, is_pyspark=True)
    elif "dbfs" in dbfs_s3_path:
        dbfs_s3_path = dbfs_s3_path.replace('/dbfs', '')
    else:
        raise ValueError("dbfs_s3_path must contain either 's3' or 'dbfs'")
    
    return dbfs_s3_path