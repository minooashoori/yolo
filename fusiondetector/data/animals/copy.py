import os
import random
import shutil

random.seed(42)

# Source and destination folder paths
# source_folder = '/home/ec2-user/dev/data/animal_dataset_v1_clean_check' # animals in general
source_folder = '/home/ec2-user/dev/data/afhq/train/cat' # dogs
destination_folder = '/home/ec2-user/dev/data/newfusion/merge/yolo_manual/images/train'

# List all .jpg files in the source folder
jpg_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]

# # Calculate the number of files to copy
num_to_copy = int(len(jpg_files) * 0.1)

# # Randomly select 10% of files
selected_files = random.sample(jpg_files, num_to_copy)


# Copy selected files to the destination folder
for file in selected_files:
    source_path = os.path.join(source_folder, file)
    destination_path = os.path.join(destination_folder, "cat_"+file)
    shutil.copy(source_path, destination_path)

print(f"Successfully copied {num_to_copy} random .jpg files to the destination folder.")
