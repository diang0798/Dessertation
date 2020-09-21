import torch
from net import AirNet
from utils import util

# Model for Experiment 1 and 2
def experiment1(num_epochs, training_loader, validation_loader, lr, cr_round, experiment):

    linear_cnn = AirNet.AirNet()
    if torch.cuda.is_available():
        linear_cnn = linear_cnn.cuda()
    optimizer = torch.optim.Adam(linear_cnn.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    train_loss = []
    validation_loss = []
    # training
    for epoch in range(num_epochs):
        loss_list = []
        for i, data in enumerate(training_loader):
            img0, img1, label = data
            img0 = img0.unsqueeze(1)
            img1 = img1.unsqueeze(1)
            img0, img1, label = util.get_variable(img0), util.get_variable(img1), util.get_variable(label)

            output_matrix = linear_cnn(img0, img1)

            label = label.view(label.size()[0], -1)
            loss = loss_func(output_matrix, label)      # MSE_transformation
            loss_list.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = sum(loss_list)/len(loss_list)
        train_loss.append(epoch_loss)
        print("round%d: Training: Epoch [%d/%d],  Loss: %.4f" % (cr_round, epoch + 1, num_epochs, epoch_loss))

        #validation
        with torch.no_grad():
            loss_list_v = []
            for i, data in enumerate(validation_loader):
                img0, img1, label = data
                img0 = img0.unsqueeze(1)
                img1 = img1.unsqueeze(1)

                img0, img1, label = util.get_variable(img0), util.get_variable(img1), util.get_variable(label)
                output_matrix = linear_cnn(img0, img1)

                label = label.view(label.size()[0], -1)
                loss = loss_func(output_matrix, label)
                loss_list_v.append(loss)

            validation_epoch_loss = sum(loss_list_v)/len(loss_list_v)
            validation_loss.append(validation_epoch_loss)
            print('validation:  loss : %.4f' % (validation_epoch_loss))

    if experiment == 0:
        torch.save(linear_cnn.state_dict(), 'model/experiment0/round' + str(cr_round) + '.pkl')

    if experiment == 1:
        torch.save(linear_cnn.state_dict(), 'model/experiment1/round' + str(cr_round) + '.pkl')
