import numpy as np


def ncc(img0, img1):
    product = np.mean((img0 - img0.mean()) * (img1 - img1.mean()))
    ncc_std = img0.std() * img1.std()
    if ncc_std == 0:
        ncc_result = 0
    else:
        ncc_result = product / ncc_std

    return ncc_result
