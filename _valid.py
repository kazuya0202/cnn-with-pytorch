import errno
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torchvision.transforms as transforms
from PIL import Image

# my package
import cnn


@dataclass
class PredictedResult:
    r"""Predicted result.

    Args:
        label (int, optional): label of classes. Defaults to -1.
        name (str, optional): class name. Defaults to 'None'.
        rate (float, optional): accuracy rate. Defaults to 0.0.
    """

    label: int = -1
    name: str = 'None'
    rate: float = 0.
# end of [class] PredictedData


@dataclass(init=False)
class ValidModel:
    """Validation model."""

    def __init__(self, load_pth_path: str, use_gpu: bool = True, in_channels: int = 3,
                 input_size: tuple = (60, 60), transform: transforms = None) -> None:
        r"""
        Args:
            load_pth_path (str): pth path for loading.
            use_gpu (bool, optional): gpu. Defaults to True.
            input_size (tuple, optional): input image size. Defaults to (60, 60).
            transform (transforms, optional): transform. Defaults to None.

        Raises:
            FileNotFoundError: file not exists.
        """
        # device
        use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_gpu else 'cpu')

        self.input_size = input_size
        self.in_channels = in_channels

        # load model
        self._load_model(load_pth_path)
        self.classify_size = len(self.classes)

        if transform is None:
            # transform
            transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.transform = transform
    # end of [function] __init__

    def _load_model(self, pth_path: str):
        # TODO: overwrite this by tu.Model.

        """Loading model.

        Args:
            pth_path (str): pth path.

        Raises:
            FileNotFoundError: file not exists.
        """
        # check exist
        if not os.path.exists(pth_path):
            err = OSError(errno.ENOENT, os.strerror(errno.ENOENT), pth_path)
            raise FileNotFoundError(err)

        # network
        self.net = cnn.Net(
            input_size=self.input_size,
            in_channels=self.in_channels)

        # load checkpoint
        checkpoint = torch.load(pth_path)

        # classes, network
        self.classes = checkpoint['classes']
        self.net.load_state_dict(checkpoint['model_state_dict'])

        self.net.to(self.device)  # switch to GPU / CPU
        self.net.eval()  # switch to eval

    def valid_multi_images(self, image_path_list: list) -> Iterator[PredictedResult]:
        # generator
        for path in image_path_list:
            yield self.valid(path)
    # end of [function] multi_valid

    def valid(self, image_path: str) -> PredictedResult:
        r"""Validing model by single image.

        Args:
            image_path (str): image path to valid.

        Raises:
            FileNotFoundError: is occured if image path is not exist.

        Returns:
            PredictedData: result of predicted.
        """
        # path is not exist -> PredictedData default value
        if not os.path.exists(image_path):
            # raise FileNotFoundError
            return PredictedResult()

        # get image as tensor
        img = self.preprocess(image_path)

        # input to model
        x = self.net(img)
        pred = torch.max(x.data, 1)[1].cpu().numpy()
        label = pred[0]

        name = self.classes[label]

        return PredictedResult(label, name)
    # end of [function] valid

    def preprocess(self, image_path: str) -> torch.Tensor:
        r"""Get image data as tensor type.

        Args:
            image_path (str): image path

        Returns:
            torch.Tensor: tensor of image data
        """
        img = Image.open(image_path)
        img = img.convert('RGB')

        # transform
        img = self.transform(img)

        # add fake dimension
        #   [unsqueeze] referene -> <https://pytorch.org/docs/stable/torch.html#torch.unsqueeze>
        img = torch.unsqueeze(img, 0)  # OR `img.unsqueeze_(0)`
        img = img.to(self.device)

        return img
    # end of [function] preprocess


if __name__ == '__main__':
    # 学習済みモデル
    pth_path = r'D:\workspace\repos\github.com\kazuya0202\cnn-with-pytorch\recognition_datasets\0_2020Feb10_21h48m40s_final.pth'
    use_gpu = True

    vm = ValidModel(pth_path, use_gpu, in_channels=3)

    # ===== example =====
    # 1枚
    img_path = './recognition_datasets/Images/crossing/crossing-samp1_3_4.jpg'
    result = vm.valid(img_path)
    print(result)

    exit()

    # 複数枚
    train_images_path = './config/unknown_used_images.txt'
    image_list = open(train_images_path).readlines()

    random.shuffle(image_list)  # シャッフル

    print('%s%s%s%s' % ('label'.center(10), 'class name'.center(15), 'acc rate'.center(8), ' path'))
    for img_path in image_list:
        img_path = img_path.strip()
        result = vm.valid(img_path)

        ss = ''
        ss += str(result.label).center(10)
        ss += str(result.name).center(15)
        ss += str(result.rate).center(8)
        ss += ' ' + Path(img_path).name

        print(ss)
