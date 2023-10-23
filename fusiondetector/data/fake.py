import numpy as np
import cv2
import os

def generate_fake_ims(n_ims, out_dir="/home/ec2-user/dev/data/fakes"):
    # generate fake images for adding to the training set
    # the images should be of the same size as the training images, i.e. 416x416 and have 3 channels we will dump them in a folder called fake_images

    for i in range(n_ims):

        # with some prob add a all back image or all white image
        p = np.random.random()
        if p < 0.3:
            if np.random.random() < 0.5:
                im = np.zeros((416,416,3), dtype=np.uint8)
            else:
                im = np.ones((416,416,3), dtype=np.uint8) * 255
        elif  0.3 <= p < 0.8:
            # add a single color image
            r = np.random.randint(0,255)
            g = np.random.randint(0,255)
            b = np.random.randint(0,255)
            im = np.ones((416,416,3), dtype=np.uint8) * np.array([r,g,b], dtype=np.uint8)
        else:
            # add a random image
            im = np.random.randint(0,255, size=(416,416,3), dtype=np.uint8)
        im_path = os.path.join(out_dir, f"fake_{i}.jpg")
        cv2.imwrite(im_path, im)
        print(f"Saved fake image to {im_path}")
        
generate_fake_ims(1000)