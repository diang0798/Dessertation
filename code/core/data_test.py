import torch
import scipy.io as scio
import numpy as np
from torch.utils.data import DataLoader
from core import data_provider
from utils import util
from net import AirNet


def testing(experiment, cr_round, norm):
    result_path = "E:/Y4/DT/data/cross_validation/experiment" + str(experiment) + "/round" + str(cr_round) + "/mat/"
    testing_set = np.load("data/testing.npy")
    testing_brains = []
    testing_labels = []
    for index in testing_set:
        testing_brains.append('sub-' + str(index) + '_T1_brain.nii.gz')
        testing_labels.append('sub-' + str(index) + '_T1_mat.mat')

    testing_data = data_provider.DataProvider(input_set1=testing_brains, input_set2=testing_labels, experiment=None)
    testing_loader = DataLoader(dataset=testing_data, shuffle=True, batch_size=1)
    test_loss = []
    abnomal = []
    diff_all = np.zeros((1,12))
    linear_cnn = AirNet.AirNet()
    if torch.cuda.is_available():
        linear_cnn = linear_cnn.cuda()
    loss_func = torch.nn.MSELoss()
    linear_cnn.load_state_dict(torch.load('model/experiment' + str(experiment) + '/round' + str(cr_round) + '.pkl'))
    linear_cnn.eval()
    with torch.no_grad():
        for i, data in enumerate(testing_loader):
            img0, img1, label, index = data
            index = "".join(index)
            img0 = img0.unsqueeze(1)
            img1 = img1.unsqueeze(1)
            img0, img1, label = util.get_variable(img0), util.get_variable(img1), util.get_variable(label)

            output_matrix = linear_cnn(img0, img1)

            if experiment != 1:
                output_matrix[0][3] *= norm
                output_matrix[0][7] *= norm
                output_matrix[0][11] *= norm

            label = label.view(label.size()[0], -1)
            loss = loss_func(output_matrix, label)
            diff = output_matrix - label
            diff_arr = diff.data.cpu().numpy()
            diff_arr = np.abs(diff_arr)
            diff_all += diff_arr
            if loss > 1:
                abnomal.append(loss)
            test_loss.append(loss)
            print('testing:' + str(index) + ':  loss : %.4f' % (loss))
            print('different = ' + str(diff_arr))

            arr = output_matrix.data.cpu().numpy()
            arr = arr.reshape(3, 4)
            arr = np.append(arr, np.array([[0, 0, 0, 1]]), 0)
            mat_name = result_path + str(index) + 'mat_result.mat'
            scio.savemat(mat_name, {'mat': arr})

        testLoss = sum(test_loss)/len(test_loss)
        print('round' + str(cr_round) + 'testMean = ' + str(testLoss))

        realLoss = (sum(test_loss) - sum(abnomal)) / (len(test_loss) - len(abnomal))
        print('realMean = ' + str(realLoss))
        print('abnomal = ' + str(len(abnomal)) + str(abnomal))
        avg_diff = diff_all / len(test_loss)
        print('diff_all = ' + str(avg_diff))