import torch
from net import AirNet
from utils import util
from utils import loss_function as loss_f
import scipy.ndimage as ndi
import numpy as np


# Model for experiment 4
def experiment4(num_epochs, training_loader, validation_loader, lr, cr_round, batch_size, norm):

    linear_cnn = AirNet.AirNet()
    if torch.cuda.is_available():
        linear_cnn = linear_cnn.cuda()
    optimizer = torch.optim.Adam(linear_cnn.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    train_loss = []
    validation_loss = []
    L3 = 0
    for epoch in range(num_epochs):
        loss_list = []
        loss_orig_list = []
        loss_L3_list = []
        for i, data in enumerate(training_loader):
            img0, img1, gt_img, label = data
            img0 = img0.unsqueeze(1)
            img1 = img1.unsqueeze(1)
            img0, img1, label = util.get_variable(img0), util.get_variable(img1), util.get_variable(label)

            output_matrix = linear_cnn(img0, img1)

            if epoch >= 15:
                transform = output_matrix.data.cpu().numpy()
                transform = transform.reshape(5, 3, 4)
                transform16 = np.zeros((5, 4, 4))
                homo = [[0, 0, 0, 1]]
                for k in range(batch_size):
                    transform16[k] = np.append(transform[k], homo, axis=0)
                    transform16[k][0][3] *= norm
                    transform16[k][1][3] *= norm
                    transform16[k][2][3] *= norm

                brain = img0.data.cpu().numpy()
                gt_img = gt_img.numpy()

                sum_ncc = []
                for j in range(batch_size):
                    matrix = np.linalg.inv(transform16[j])
                    matrix = torch.from_numpy(matrix)
                    transform_input = brain[j].squeeze(0)
                    zero_norm = transform_input[0][0][0]
                    warp_img = ndi.affine_transform(transform_input, matrix, cval=zero_norm)

                    # product = np.mean((warp_img - warp_img.mean()) * (gt_img[j] - gt_img[j].mean()))
                    # ncc_std = warp_img.std() * gt_img[j].std()
                    # if ncc_std == 0:
                    #     ncc = 0
                    # else:
                    #     ncc = product / ncc_std
                    ncc = loss_f.ncc(warp_img, gt_img)
                    sum_ncc.append(ncc)
                if (epoch + 1) % 20 == 0:
                    print(sum_ncc)
                avg_ncc = sum(sum_ncc) / len(sum_ncc)
                L3 = 1 - ((avg_ncc + 1) / 2)
                L3 = torch.tensor(L3)
                L3 *= 0.1

            label = label.view(label.size()[0], -1)
            L1 = loss_func(output_matrix, label)
            loss_orig_list.append(L1)

            loss = L1 + L3   # MSE_transformation + NCC_similarity

            loss_list.append(loss)
            loss_L3_list.append(L3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = sum(loss_list) / len(loss_list)
        epoch_loss_L3 = sum(loss_L3_list) / len(loss_L3_list)
        epoch_loss_orig = sum(loss_orig_list) / len(loss_orig_list)
        train_loss.append(epoch_loss)
        print("round%d: Training: Epoch [%d/%d],  Loss: %.4f" % (cr_round, epoch + 1, num_epochs, epoch_loss))
        print('loss_orig =' + str(epoch_loss_orig))
        print('loss_L3 =' + str(epoch_loss_L3))

        # validation
        with torch.no_grad():
            loss_list_v = []
            loss_orig_list_v = []
            loss_L3_list_v = []
            for i, data in enumerate(validation_loader):
                img0, img1, gt_img, label = data
                img0 = img0.unsqueeze(1)
                img1 = img1.unsqueeze(1)

                img0, img1, label = util.get_variable(img0), util.get_variable(img1), util.get_variable(label)
                output_matrix = linear_cnn(img0, img1)

                if epoch >= 15:
                    transform = output_matrix.data.cpu().numpy()
                    transform = transform.reshape(5, 3, 4)
                    transform16 = np.zeros((5, 4, 4))
                    homo = [[0, 0, 0, 1]]
                    for k in range(batch_size):
                        transform16[k] = np.append(transform[k], homo, axis=0)
                        transform16[k][0][3] *= norm
                        transform16[k][1][3] *= norm
                        transform16[k][2][3] *= norm

                    brain = img0.data.cpu().numpy()
                    gt_img = gt_img.numpy()

                    sum_ncc = []
                    for j in range(batch_size):
                        matrix = np.linalg.inv(transform16[j])
                        matrix = torch.from_numpy(matrix)
                        transform_input = brain[j].squeeze(0)
                        zero_norm = transform_input[0][0][0]
                        warp_img = ndi.affine_transform(transform_input, matrix, zero_norm)
                        ncc = loss_f.ncc(warp_img, gt_img)
                        sum_ncc.append(ncc)
                    avg_ncc = sum(sum_ncc) / len(sum_ncc)
                    L3 = 1 - ((avg_ncc + 1) / 2)
                    L3 = torch.tensor(L3)
                    L3 *= 0.1

                label = label.view(label.size()[0], -1)
                L1 = loss_func(output_matrix, label)
                loss_orig_list_v.append(L1)

                loss = L1 + L3

                loss_list_v.append(loss)
                loss_L3_list_v.append(L3)

            validation_epoch_loss = sum(loss_list_v) / len(loss_list_v)
            epoch_loss_L3_v = sum(loss_L3_list_v) / len(loss_L3_list_v)
            epoch_loss_orig_v = sum(loss_orig_list_v) / len(loss_orig_list_v)
            validation_loss.append(validation_epoch_loss)
            print('validation:  loss : %.4f' % (validation_epoch_loss))
            print('loss_orig =' + str(epoch_loss_orig_v))
            print('loss_L3 =' + str(epoch_loss_L3_v))

    torch.save(linear_cnn.state_dict(), 'model/experiment4/round' + str(cr_round) + '.pkl')