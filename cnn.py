import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Net(nn.Module):
    # def __init__(self, input_size: tuple, classify_size: int, in_channels: int = 3) -> None:
    def __init__(self, **options) -> None:
        r"""
        **options
            input_size (tuple): Defaults to (60, 60).
            classify_size (int): Defaults to 3.
            in_channels (int): Defaults to 3.
        """
        super(Net, self).__init__()

        # options
        input_size: tuple = options.pop('input_size', (60, 60))
        classify_size: int = options.pop('classify_size', 3)
        in_channels: int = options.pop('in_channels', 3)

        hei_ = input_size[0] // 4  # `4` depends on max_pool2d.
        wid_ = input_size[1] // 4

        # -- using parameters of pytorch module --
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.fc6 = nn.Linear((256 * hei_ * wid_), 2048)
        self.fc7 = nn.Linear(2048, 512)
        self.fc8 = nn.Linear(512, classify_size)
    # end of [function] __init__

    def forward(self, x) -> Tensor:
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, num_flat_features(x))  # resize tensor
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)

        # don't run `softmax()` because of softmax process in CrossEntropyLoss
        # x = F.softmax(x)

        return x
    # end of [function] forward
# end of [class] Net


def num_flat_features(x) -> int:
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features
# end of [function] num_flat_features
