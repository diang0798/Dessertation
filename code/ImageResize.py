import matplotlib
matplotlib.use('TkAgg')
import nibabel as nib
import numpy as np
import math
import scipy.ndimage as ndi
import os

brain_path = "E:/Y4/DT/data/testing/brains"
resized_path = "E:/Y4/DT/data/testing/resized_64/"
files = os.listdir(brain_path)  # name of all the files in the folder
IMAGE_HEIGHT_HALF = 128
IMAGE_LENGTH_HALF = 128
IMAGE_WIDTH_HALF = 128

# go through all the images and pad them to 256*256*256
for file in files:
    img = nib.load(brain_path + "/" + file)
    # img = nib.load("E:/Y4/DT/data/MNI152_T1_1mm_brain.nii.gz")
    img_data = img.get_data()
    img_arr = img.dataobj[:, :, :].copy()
    print(img_arr.shape )
    x, y, z = img.shape
    print(str(x) + " " + str(y) + " " + str(z))

    if x < IMAGE_WIDTH_HALF * 2:
        img_arr = np.pad(img_arr, ((IMAGE_WIDTH_HALF - math.ceil(x / 2), IMAGE_WIDTH_HALF - math.floor(x / 2)),
                                   (0, 0), (0, 0)), 'constant')
    else:
        x_start = round(x / 2) - IMAGE_WIDTH_HALF
        x_end = round(x / 2) + IMAGE_WIDTH_HALF
        img_arr = img_arr[x_start:x_end, :, :]

    if y < IMAGE_LENGTH_HALF * 2:
        img_arr = np.pad(img_arr,
                         ((0, 0), (IMAGE_LENGTH_HALF - math.ceil(y / 2), IMAGE_LENGTH_HALF - math.floor(y / 2)),
                          (0, 0)), 'constant')
    else:
        y_start = round(y / 2) - IMAGE_LENGTH_HALF
        y_end = round(y / 2) + IMAGE_LENGTH_HALF
        img_arr = img_arr[:, y_start:y_end, :]

    if z < IMAGE_HEIGHT_HALF * 2:
        img_arr = np.pad(img_arr, ((0, 0), (0, 0),
                                   (IMAGE_HEIGHT_HALF - math.ceil(z / 2), IMAGE_HEIGHT_HALF - math.floor(z / 2))),
                         'constant')
    else:
        z_start = round(z / 2) - IMAGE_HEIGHT_HALF
        z_end = round(z / 2) + IMAGE_HEIGHT_HALF
        img_arr = img_arr[:, :, z_start:z_end]

    print("shape = " + str(img_arr.shape))
    img_arr = ndi.zoom(img_arr, 0.25)   # resize the image from 256*256*256 to 64*64*64
    print("shape = " + str(img_arr.shape))
    nib_img = nib.Nifti1Image(img_arr, img.affine)
    nib.save(nib_img, resized_path + file)
    # nib.save(nib_img, "E:/Y4/DT/data/Resized64_MNI152_T1_1mm_brain.nii.gz")
