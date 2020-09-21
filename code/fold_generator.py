import os
import numpy as np
import re

brain_path = "E:/Y4/DT/data/resized_brain_64/"
fold = 3
num_fold = 90

brain_list = os.listdir(brain_path)
file_index_list = []
for file_name in brain_list:
    pattern = re.compile(r'\d{5}')
    if pattern.findall(file_name):
        file_index = pattern.findall(file_name)[0]
        if file_index not in file_index_list:
            file_index_list.append(file_index)

np.random.shuffle(file_index_list)

fold1 = []
fold2 = []
fold3 = []
testing = []

fold1 = file_index_list[0: num_fold]
fold2 = file_index_list[num_fold: (2 * num_fold)]
fold3 = file_index_list[2 * num_fold: 3 * num_fold]
testing = file_index_list[3 * num_fold:]

np.save("data/fold1.npy", fold1)
np.save("data/fold2.npy", fold2)
np.save("data/fold3.npy", fold3)
np.save("data/testing.npy", testing)
print(len(file_index_list))
print(len(fold1))
print(len(fold2))
print(len(fold3))
print(len(testing))
