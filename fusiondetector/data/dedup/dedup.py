from imagededup.methods import PHash
import os

def dedup(image_dir: str, label_dir: str= None):
    """
    Remove duplicate images from the specified directory.

    Args:
        image_dir (str): The directory containing the images to deduplicate.

    Returns:
        None
    """
    # Initialize the perceptual hash generator
    phasher = PHash()

    # Generate encodings for all images in the image directory
    encodings = phasher.encode_images(image_dir=image_dir)
    duplicates = phasher.find_duplicates(encoding_map=encodings)


    # Create a set to store images to keep
    images_to_keep = set()
    images_to_remove = set()


    for key, duplicate_list in duplicates.items():
        # Check if the key is not in images_to_keep and if there are no duplicates
        if key not in images_to_keep and key not in images_to_remove:
            # Add the key image to the images to keep
            images_to_keep.add(key)
            # Add the duplicates to the images to remove
            images_to_remove.update(duplicate_list)

    print(f"Number of images to keep: {len(images_to_keep)}")
    print(f"Number of images to remove: {len(images_to_remove)}")

    img_paths_to_remove = [os.path.join(image_dir, path) for path in images_to_remove]
    label_paths_to_remove = [os.path.join(label_dir, path.replace(".jpg", ".txt")) for path in images_to_remove] if label_dir is not None else None

c
    # Remove the duplicate images/labels
    for file_path in img_paths_to_remove:
        os.remove(file_path)
    if label_paths_to_remove:
        for file_path in label_paths_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == "__main__":
    # Example usage
    dedup('/home/ec2-user/dev/data/portal_copy/yolo/images/train', '/home/ec2-user/dev/data/portal_copy/yolo/labels/train')