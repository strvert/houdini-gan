import torch
from torch import nn


class CNNNetMinimum(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(16 * 15 * 15, 120)
        self.fc_out = nn.Linear(120, 47)

        self.relu = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(-1, 16 * 15 * 15)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x

    @staticmethod
    def network_name():
        return "cnn-minimum"


class CNNNetBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_outs = 128

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.conv_outs, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=self.conv_outs, out_channels=self.conv_outs, kernel_size=3, stride=1,
                               padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(self.conv_outs * 8 * 8, 120)
        self.fc_out = nn.Linear(120, 47)

        self.relu = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(-1, self.conv_outs * 8 * 8)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x

    @staticmethod
    def network_name():
        return "cnn-basic"


class CNNNetWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_outs = 128

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.conv_outs, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=self.conv_outs, out_channels=self.conv_outs, kernel_size=3, stride=1,
                               padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(self.conv_outs * 8 * 8, 120)
        self.fc_out = nn.Linear(120, 47)

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(self.conv_outs)
        self.batchnorm2 = nn.BatchNorm2d(self.conv_outs)
        self.batchnorm3 = nn.BatchNorm1d(120)

    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(-1, self.conv_outs * 8 * 8)
        x = self.fc(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x

    @staticmethod
    def network_name():
        return "cnn-with-bn"
