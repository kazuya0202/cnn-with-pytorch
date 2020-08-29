import errno
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Tuple

import torch
import torch.nn.functional as F
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
    name: str = "None"
    rate: float = 0.0


@dataclass
class ValidModel:
    r"""Validation model.
    Args:
        load_pth_path (str): pth path for loading.
        use_gpu (bool, optional): gpu. Defaults to True.
        input_size (tuple, optional): input image size. Defaults to (60, 60).
        transform (transforms, optional): transform. Defaults to None.
    """
    load_pth_path: str
    use_gpu: bool = True
    in_channels: int = 3
    input_size: Tuple[int, int] = (60, 60)
    transform: Any = None

    def __post_init__(self):
        r"""
        Raises:
            FileNotFoundError: file not exists.
        """
        # device
        self.use_gpu = self.use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        self.classify_size: int

        # load model
        self._load(self.load_pth_path)

        if self.transform is None:
            # transform
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

    def _load(self, pth_path: str):
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

        # load checkpoint
        checkpoint = torch.load(pth_path)

        # classes, network
        self.classes = checkpoint["classes"]
        self.classify_size = len(self.classes)

        # network
        self.net = cnn.Net(
            input_size=self.input_size,
            classify_size=self.classify_size,
            in_channels=self.in_channels,
        )
        self.net.load_state_dict(checkpoint["model_state_dict"])

        self.net.to(self.device)  # switch to GPU / CPU
        self.net.eval()  # switch to eval

    def valid_multi_images(self, image_path_list: list) -> Iterator[PredictedResult]:
        # generator
        for path in image_path_list:
            yield self.valid(path)

    def valid(self, image_path: str) -> PredictedResult:
        r"""Validing model by single image.

        Args:
            image_path (str): image path to valid.

        Returns:
            PredictedData: result of predicted.
        """
        # path is not exist -> PredictedData default value
        if not os.path.exists(image_path):
            return PredictedResult()

        with torch.no_grad():
            # get image as tensor
            img = self.preprocess(image_path)

            # input to model
            x: torch.Tensor = self.net(img)
            # pred = torch.max(x.data, 1)[1].cpu().numpy()

            x_sm = F.softmax(x, -1)
            pred = torch.max(x_sm.data, 1)

            label = int(pred[1].item())  # label num
            name = self.classes[label]  # class name
            acc_rate = float(pred[0].item())  # accuracy score (#beta)

        return PredictedResult(label, name, acc_rate)

    def preprocess(self, image_path: str) -> torch.Tensor:
        r"""Get image data as tensor type.

        Args:
            image_path (str): image path

        Returns:
            torch.Tensor: tensor of image data
        """
        img = Image.open(image_path)
        img = img.convert("RGB")

        # transform
        img = self.transform(img)

        # add fake dimension
        #   [unsqueeze] referene -> <https://pytorch.org/docs/stable/torch.html#torch.unsqueeze>
        img = torch.unsqueeze(img, 0)  # OR `img.unsqueeze_(0)`
        img = img.to(self.device)

        return img


if __name__ == "__main__":
    # 学習済みモデル
    pth_path = r"D:\workspace\repos\github.com\kazuya0202\cnn-with-pytorch\recognition_datasets\0_2020Feb10_21h48m40s_final.pth"
    use_gpu = True

    vm = ValidModel(pth_path, use_gpu, in_channels=3)

    # ===== example =====
    # 1枚
    img_path = "./recognition_datasets/Images/crossing/crossing-samp1_0_1.jpg"
    result = vm.valid(img_path)
    print(result)

    exit()

    # 複数枚
    train_images_path = "./config/unknown_used_images.txt"
    image_list = open(train_images_path).readlines()

    random.shuffle(image_list)  # シャッフル

    print("%s%s%s%s" % ("label".center(10), "class name".center(15), "acc rate".center(8), " path"))
    for img_path in image_list:
        img_path = img_path.strip()
        result = vm.valid(img_path)

        ss = ""
        ss += str(result.label).center(10)
        ss += str(result.name).center(15)
        ss += str(result.rate).center(8)
        ss += " " + Path(img_path).name

        print(ss)
