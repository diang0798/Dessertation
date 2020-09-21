import torch
import scipy.ndimage as ndi
import numpy as np
import random


def img_aug(brain_img, norm, zero_norm):
    matrix = np.eye(4, 4)
    shift = 5
    translation_x = random.uniform(-shift, shift)
    translation_y = random.uniform(-shift, shift)
    translation_z = random.uniform(-shift, shift)
    matrix[0][3] = translation_x / norm
    matrix[1][3] = translation_y / norm
    matrix[2][3] = translation_z / norm

    scale_b = 0.9
    scale_t = 1.1
    scale_x = random.uniform(scale_b, scale_t)
    scale_y = random.uniform(scale_b, scale_t)
    scale_z = random.uniform(scale_b, scale_t)
    scale_matrix = np.array([[scale_x, 0, 0, 0], [0, scale_y, 0, 0], [0, 0, scale_z, 0], [0, 0, 0, 1]])
    matrix = np.dot(matrix, scale_matrix)

    degrees = 2
    degree_x = random.uniform(-degrees, degrees)
    degree_y = random.uniform(-degrees, degrees)
    degree_z = random.uniform(-degrees, degrees)
    radian_x = -np.pi * degree_x / 180.
    radian_y = -np.pi * degree_y / 180.
    radian_z = -np.pi * degree_z / 180.
    rotation_x = np.array(
        [[1, 0, 0, 0], [0, np.cos(radian_x), np.sin(radian_x), 0], [0, -np.sin(radian_x), np.cos(radian_x), 0],
         [0, 0, 0, 1]])
    rotation_y = np.array(
        [[np.cos(radian_y), 0, np.sin(radian_y), 0], [0, 1, 0, 0], [-np.sin(radian_y), 0, np.cos(radian_y), 0],
         [0, 0, 0, 1]])
    rotation_z = np.array(
        [[np.cos(radian_z), np.sin(radian_z), 0, 0], [-np.sin(radian_z), np.cos(radian_z), 0, 0], [0, 0, 1, 0],
         [0, 0, 0, 1]])
    rotation = np.dot(rotation_x, np.dot(rotation_y, rotation_z))

    matrix = np.dot(matrix, rotation)
    matrix = np.float32(matrix)

    matrix_inv = np.linalg.inv(matrix)
    matrix = torch.from_numpy(matrix)

    aug_img = ndi.affine_transform(brain_img, matrix_inv, cval=zero_norm)
    aug_img = torch.from_numpy(aug_img)

    return matrix, aug_img
