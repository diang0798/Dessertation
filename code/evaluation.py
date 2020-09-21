import os
import nibabel as nib
from skimage.measure import compare_mse as mse
import numpy as np
from utils import loss_function as loss_f


current_experiment = 5

ground_truth_path = "E:/Y4/DT/data/cross_validation/testing_set/affined_brains"
ground_truth_list = os.listdir(ground_truth_path)

template_path = "E:/Y4/DT/data/Resized64_MNI152_T1_1mm_brain.nii.gz"
template_img = nib.load(template_path)
template = template_img.get_data()
min_val_t = np.min(template)
max_val_t = np.max(template)
template_norm = (template - min_val_t) / (max_val_t - min_val_t)


# Evaluate original image and ground truth
brain_path = "E:/Y4/DT/data/cross_validation/testing_set/brains"
brain_list = os.listdir(brain_path)
orig_mse = []
orig_ncc = []
gt_mse = []
gt_ncc = []
for k in range(len(brain_list)):
    brain_img = nib.load(brain_path + "/" + brain_list[k]).get_data()
    min_val = np.min(brain_img)
    max_val = np.max(brain_img)
    brain_norm = (brain_img - min_val) / (max_val - min_val)

    mse_brain = mse(brain_norm, template_norm)
    orig_mse.append(mse_brain)
    ncc_brain = loss_f.ncc(brain_norm, template_norm)
    orig_ncc.append(ncc_brain)

    ground_truth_img = nib.load(ground_truth_path + "/" + ground_truth_list[k]).get_data()
    min_val = np.min(ground_truth_img)
    max_val = np.max(ground_truth_img)
    ground_truth_norm = (ground_truth_img - min_val) / (max_val - min_val)

    mse_standard = mse(ground_truth_norm, template_norm)
    gt_mse.append(mse_standard)
    ncc0 = loss_f.ncc(ground_truth_norm, template_norm)
    gt_ncc.append(ncc0)

brain_mse = np.mean(orig_mse)
brain_ncc = np.mean(orig_ncc)
gt_mse_mean = np.mean(gt_mse)
gt_ncc_mean = np.mean(gt_ncc)
np.array(orig_mse)
np.array(orig_ncc)
np.array(gt_mse)
np.array(gt_ncc)
brain_mse_std = np.std(orig_mse)
brain_ncc_std = np.std(orig_ncc)
gt_mse_std = np.std(gt_mse)
gt_ncc_std = np.std(gt_ncc)
print("orig_mse = " + str(brain_mse))
print("orig_ncc = " + str(brain_ncc))
print("orig_mse_std = " + str(brain_mse_std))
print("orig_ncc_std = " + str(brain_ncc_std))
print("gt_mse = " + str(gt_mse_mean))
print("gt_ncc = " + str(gt_ncc_mean))
print("gt_mse_std = " + str(gt_mse_std))
print("gt_ncc_std = " + str(gt_ncc_std))


# Evaluate the result of current experiment
for i in range(3):
    affined_result_path = "E:/Y4/DT/data/cross_validation/experiment" + str(current_experiment) + "/round" + str(i) + "/affined"
    k = 0
    result_mse = []
    ncc_my = []
    for j in range(len(ground_truth_list)):
        affined_result_img = nib.load(affined_result_path + "/" + ground_truth_list[j]).get_data()
        min_val = np.min(affined_result_img)
        max_val = np.max(affined_result_img)
        affined_result_norm = (affined_result_img - min_val) / (max_val - min_val)

        mse_result = mse(affined_result_norm, template_norm)
        result_mse.append(mse_result)
        ncc1 = loss_f.ncc(affined_result_norm, template_norm)
        ncc_my.append(ncc1)

    result_mean = np.mean(result_mse)
    ncc1_mean = np.mean(ncc_my)
    np.array(result_mse)
    np.array(ncc1_mean)
    mse_std = np.std(result_mse)
    ncc_std = np.std(ncc_my)
    print("round" + str(i))
    print("result_mse = " + str(result_mean))
    print("mean_result = " + str(ncc1_mean))
    print("mse_std = " + str(mse_std))
    print("ncc_std = " + str(ncc_std))
