from fusiondetector.data.merge import MergeImgDataset

merge = MergeImgDataset(
    datasetA_path="/home/ec2-user/dev/data/newfusion/merge/yolo_manual",
    datasetB_path="/home/ec2-user/dev/data/portal/yolo",
    output_folder="/home/ec2-user/dev/data/newfusion_portal")

merge.merge(splits=["train", "val"], includeA=["train", "val"], includeB=["train"])


