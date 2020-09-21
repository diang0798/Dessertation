import torch
from net import AirNet
from utils import util
import numpy as np


# Model for experiment 5
def experiment5(num_epochs, training_loader, validation_loader, lr, cr_round, batch_size):

    linear_cnn = AirNet.AirNet()
    if torch.cuda.is_available():
        linear_cnn = linear_cnn.cuda()
    optimizer = torch.optim.Adam(linear_cnn.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    train_loss = []
    validation_loss = []
    for epoch in range(num_epochs):
        loss_list = []
        loss_orig_list = []
        loss_augt_list = []
        for i, data in enumerate(training_loader):
            img0, img1, label, Ta_img, Ta_label = data
    
            img0, img1, label = util.get_variable(img0), util.get_variable(img1), util.get_variable(label)
            img0 = img0.unsqueeze(1)
            img1 = img1.unsqueeze(1)
            Ta_img = util.get_variable(Ta_img)
            Ta_img = Ta_img.unsqueeze(1)

            output_matrix = linear_cnn(img0, img1)
            output_matrix_ta = linear_cnn(Ta_img, img1)

            ta12 = output_matrix_ta.data.cpu().numpy()
            ta12 = ta12.reshape(5, 3, 4)
            ta16 = np.zeros((5, 4, 4))
            dot_ta = np.zeros((5, 3, 4))
            homo = [[0, 0, 0, 1]]
            for k in range(batch_size):
                ta16[k] = np.append(ta12[k], homo, axis=0)
    
            for j in range(batch_size):
                dot_ta[j] = np.dot(Ta_label[j], ta16[j])[0:3, :]
            dot_ta = np.float32(dot_ta)
            dot_ta = torch.from_numpy(dot_ta)
            dot_ta = util.get_variable(dot_ta)

            label = label.view(label.size()[0], -1)
            dot_ta = dot_ta.view(dot_ta.size()[0], -1)

            L1 = loss_func(output_matrix, label)
            L2 = loss_func(output_matrix, dot_ta)

            loss = L1 + L2
            loss_list.append(loss)
            loss_orig_list.append(L1)
            loss_augt_list.append(L2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = sum(loss_list)/len(loss_list)
        epoch_loss_augt = sum(loss_augt_list) / len(loss_augt_list)
        epoch_loss_orig = sum(loss_orig_list) / len(loss_orig_list)
        train_loss.append(epoch_loss)
        print("round%d: Training: Epoch [%d/%d],  Loss: %.4f" % (cr_round, epoch + 1, num_epochs, epoch_loss))
        print('loss_orig =' + str(epoch_loss_orig))
        print('loss_augt =' + str(epoch_loss_augt))
    
        #validation
        with torch.no_grad():
            loss_list_v = []
            loss_orig_list_v = []
            loss_augt_list_v = []
            for i, data in enumerate(validation_loader):
                img0, img1, label, Ta_img, Ta_label = data
    
                img0, img1, label = util.get_variable(img0), util.get_variable(img1), util.get_variable(label)
                img0 = img0.unsqueeze(1)
                img1 = img1.unsqueeze(1)
                Ta_img = util.get_variable(Ta_img)
                Ta_img = Ta_img.unsqueeze(1)
    
                output_matrix = linear_cnn(img0, img1)
                output_matrix_ta = linear_cnn(Ta_img, img1)
    
                ta12 = output_matrix_ta.data.cpu().numpy()
                ta12 = ta12.reshape(5, 3, 4)
                ta16 = np.zeros((5, 4, 4))
                dot_ta = np.zeros((5, 3, 4))
                homo = [[0, 0, 0, 1]]
                for k in range(batch_size):
                    ta16[k] = np.append(ta12[k], homo, axis=0)
    
                for j in range(batch_size):
                    dot_ta[j] = np.dot(Ta_label[j], ta16[j])[0:3, :]
                dot_ta = np.float32(dot_ta)
                dot_ta = torch.from_numpy(dot_ta)
                dot_ta = util.get_variable(dot_ta)
    
                label = label.view(label.size()[0], -1)
                dot_ta = dot_ta.view(dot_ta.size()[0], -1)
    
                L1 = loss_func(output_matrix, label)
                L2 = loss_func(output_matrix, dot_ta)

                loss = L1 + L2   # MSE_transformation + MSE_consistency
    
                loss_list_v.append(loss)
                loss_orig_list_v.append(L1)
                loss_augt_list_v.append(L2)
    
            validation_epoch_loss = sum(loss_list_v) / len(loss_list_v)
            epoch_loss_augt_v = sum(loss_augt_list_v) / len(loss_augt_list_v)
            epoch_loss_orig_v = sum(loss_orig_list_v) / len(loss_orig_list_v)
            validation_loss.append(validation_epoch_loss)
            print('validation:  loss : %.4f' % (validation_epoch_loss))
            print('loss_orig =' + str(epoch_loss_orig_v))
            print('loss_augt =' + str(epoch_loss_augt_v))

    torch.save(linear_cnn.state_dict(), 'model/experiment5/round' + str(cr_round) + '.pkl')