import torch
import torch.nn as nn


# 3x3x3 conv
def conv3x3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size= 3,
                     stride=stride, bias=False, padding=1)


class AirNet(nn.Module):
    def __init__(self):
        super(AirNet, self).__init__()
        self.cnn1 = torch.nn.Sequential(
            conv3x3x3(in_channels=1, out_channels=2, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            conv3x3x3(2, 4, 1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            conv3x3x3(4, 8, 1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            conv3x3x3(8, 16, 1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(2 * 16 * 4 * 4 * 4, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.Linear(128, 12),
        )

    def forward_conv(self, x):
        output_cnn = self.cnn1(x)

        return output_cnn

    def forward(self, moving_image, fixed_image):
        output1 = self.forward_conv(moving_image)
        output2 = self.forward_conv(fixed_image)

        arr1 = output1.view(output1.size()[0], -1)
        arr2 = output2.view(output2.size()[0], -1)
        in_fc = torch.cat((arr1, arr2), 1)

        output = self.fc1(in_fc)

        return output
