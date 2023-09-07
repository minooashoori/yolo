from reader import ReadProcessBoxes


preprocess_logo05_test = ReadProcessBoxes(
    filepath_combined="/dbfs/mnt/innovation/pdacosta/data/logo05/test/",
    filepath_categories_map="/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/map/config.yaml",
    input_format="parquet",
    save_path="/dbfs/mnt/innovation/pdacosta/data/logo05/annotations/test/",
)

preprocess_logo05_test.read()


preprocess_logo05_train = ReadProcessBoxes(
    filepath_combined="/dbfs/mnt/innovation/pdacosta/data/logo05/train/",
    filepath_categories_map="/dbfs/mnt/innovation/pdacosta/data/total_fusion_02/map/config.yaml",
    input_format="parquet",
    save_path="/dbfs/mnt/innovation/pdacosta/data/logo05/annotations/train/",
)

preprocess_logo05_train.read()