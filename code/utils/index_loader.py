import numpy as np
from torch.utils.data import DataLoader
from core import data_provider


def dataset_generator(norm, experiment, train_batch_size, cr_round):
    fold1 = np.load("data/fold1.npy")
    fold2 = np.load("data/fold2.npy")
    fold3 = np.load("data/fold3.npy")

    fold1_brains = []
    fold1_labels = []
    fold1_affined = []
    fold2_brains = []
    fold2_labels = []
    fold2_affined = []
    fold3_brains = []
    fold3_labels = []
    fold3_affined = []
    for index in fold1:
        fold1_brains.append('sub-' + str(index) + '_T1_brain.nii.gz')
        fold1_labels.append('sub-' + str(index) + '_T1_mat.mat')
        fold1_affined.append('sub-' + str(index) + '_T1_affined_brain.nii.gz')

    for index in fold2:
        fold2_brains.append('sub-' + str(index) + '_T1_brain.nii.gz')
        fold2_labels.append('sub-' + str(index) + '_T1_mat.mat')
        fold2_affined.append('sub-' + str(index) + '_T1_affined_brain.nii.gz')

    for index in fold3:
        fold3_brains.append('sub-' + str(index) + '_T1_brain.nii.gz')
        fold3_labels.append('sub-' + str(index) + '_T1_mat.mat')
        fold3_affined.append('sub-' + str(index) + '_T1_affined_brain.nii.gz')

    training_brains = []
    training_labels = []
    validation_brains = []
    validation_labels = []

    if cr_round == 0:
        training_brains = fold1_brains + fold2_brains
        training_labels = fold1_labels + fold2_labels
        training_affined = fold1_affined + fold2_affined
        validation_brains = fold3_brains
        validation_labels = fold3_labels
        validation_affined = fold3_affined
    elif cr_round == 1:
        training_brains = fold2_brains + fold3_brains
        training_labels = fold2_labels + fold3_labels
        training_affined = fold2_affined + fold3_affined
        validation_brains = fold1_brains
        validation_labels = fold1_labels
        validation_affined = fold1_affined
    elif cr_round == 2:
        training_brains = fold1_brains + fold3_brains
        training_labels = fold1_labels + fold3_labels
        training_affined = fold1_affined + fold3_affined
        validation_brains = fold2_brains
        validation_labels = fold2_labels
        validation_affined = fold2_affined

    if experiment == 4:
        training_data = data_provider.DataProvider(input_set1=training_brains, input_set2=training_labels,
                                                   input_set4=training_affined, norm=norm, experiment=experiment)
        training_loader = DataLoader(dataset=training_data, shuffle=True, batch_size=train_batch_size)
        validation_data = data_provider.DataProvider(input_set1=validation_brains, input_set2=validation_labels,
                                                     input_set4=validation_affined, norm=norm, experiment=experiment)
        validation_loader = DataLoader(dataset=validation_data, shuffle=True, batch_size=train_batch_size)
    else:
        training_data = data_provider.DataProvider(input_set1=training_brains, input_set2=training_labels, norm=norm,
                                                   experiment=experiment)
        training_loader = DataLoader(dataset=training_data, shuffle=True, batch_size=train_batch_size)
        validation_data = data_provider.DataProvider(input_set1=validation_brains, input_set2=validation_labels,
                                                     norm=norm, experiment=experiment)
        validation_loader = DataLoader(dataset=validation_data, shuffle=True, batch_size=train_batch_size)

    return training_loader, validation_loader
