from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()

        hei_input = input_size[0] // 4
        wid_input = input_size[1] // 4

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=96,
            kernel_size=7,
            padding=3,
            stride=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=128,
            padding=2,
            stride=1,
            kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            padding=1,
            stride=1,
            kernel_size=3)
        self.conv4 = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            padding=1,
            stride=1,
            kernel_size=3)
        self.conv5 = nn.Conv2d(
            in_channels=384,
            out_channels=256,
            padding=1,
            stride=1,
            kernel_size=3)

        # self.fc6 = nn.Linear(256 * 16 * 16, 2048)

        # self.fc6 = nn.Linear(256 * in_height * in_width, 1024)
        # self.fc7 = nn.Linear(1024, 512)
        # self.fc8 = nn.Linear(512, 3)
        self.fc6 = nn.Linear(256 * hei_input * wid_input, 2048)
        self.fc7 = nn.Linear(2048, 512)
        self.fc8 = nn.Linear(512, 3)

        self.features = nn.ModuleDict(OrderedDict({
            'conv1': self.conv1,
            'relu1': nn.ReLU(),
            'bn1': self.bn1,
            'conv2': self.conv2,
            'relu2': nn.ReLU(),
            'bn2': self.bn2,
            'pool1': nn.MaxPool2d(2),
            'conv3': self.conv3,
            'relu3': nn.ReLU(),
            'conv4': self.conv4,
            'relu4': nn.ReLU(),
            'conv5': self.conv5,
            'relu5': nn.ReLU(),
            'pool2': nn.MaxPool2d(2)
        }))

        self.classifier = nn.Sequential(
            self.fc6,
            nn.ReLU(),
            self.fc7,
            nn.ReLU(),
            self.fc8
        )
    # end of [function] __init__

    # from pytorch_memlab import profile
    # @profile
    def forward(self, x):

        # x = F.relu(self.conv1(x))
        # x = self.bn1(x)
        # x = F.relu(self.conv2(x))
        # x = self.bn2(x)
        # x = F.max_pool2d(x, 2)
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.max_pool2d(x, 2)

        # self.fc6 の input 計算用
        # print(x.size())

        x = self.features(x)
        x = x.view(-1, num_flat_features(x))
        x = self.classifier(x)

        # x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        # x = self.fc8(x)

        # don't run `softmax()` because of softmax process in CrossEntropyLoss
        # x = F.softmax(x)

        return x
    # end of [function] forward

    # end of [function] num_flat_features
# end of [class] Net


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features
