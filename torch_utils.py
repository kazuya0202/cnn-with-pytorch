import itertools
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorboardX as tbx
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

# my packages
import cnn
import toml_settings as _tms
import utils as ul
from grad_cam import ExecuteGradCAM

# quote package
from radam_optim import radam


@dataclass
class Data:
    r"""Nesessary datas of create dataset."""
    path: str  # image path
    label: int  # label of class
    name: str  # class name

    def items(self):
        r"""Returns items of Data class.

        Returns:
            tuple: Items of class.
        """
        return self.path, self.label, self.name


@dataclass
class CreateDataset(Dataset):
    r"""Creating dataset.

    Args:
        path (str): path of image directory.
        extensions (list): supported extensions.
        test_size (Union[float, int]): test image size.
        config_path (str, optional): export path of config. Defaults to 'config'.
    """
    path: str
    extensions: list
    test_size: Union[float, int]
    config_path: str = "config"
    limit_size: Optional[int] = None

    def __post_init__(self):
        # self.path = path
        # self.extensions = extensions
        # self.test_size = test_size
        # self.config_path = config_path
        # self.limit_size = limit_size

        # {'train': [], 'unknown': [], 'known': []}
        self.all_list: Dict[str, List[Data]]

        # {label: 'class name' ...}
        self.classes: Dict[int, str]

        # size of images
        self.all_size: int  # train_size + unknown_size
        self.train_size: int
        self.unknown_size: int
        self.known_size: int

        # ----------

        self._get_all_datas()  # train / unknown / known
        self._write_config()  # write config of model

    def _get_all_datas(self) -> None:
        r"""Get all datas from each directory."""

        # init
        self.all_list = {"train": [], "unknown": [], "known": []}
        self.classes = {}

        path = Path(self.path)

        # directories in [image_path]
        dirs = [d for d in path.glob("*") if d.is_dir()]

        # all extensions / all sub directories
        for label_idx, _dir in enumerate(dirs):
            xs = []

            for ext in self.extensions:
                tmp = [
                    Data(x.as_posix(), label_idx, _dir.name)
                    for x in _dir.glob(f"*.{ext}")
                    if x.is_file()
                ]
                xs.extend(tmp)

            # adjust to limit size
            if self.limit_size is not None:
                random.shuffle(xs)
                xs = xs[: self.limit_size]

            # split dataset
            train, test = train_test_split(xs, test_size=self.test_size, shuffle=True)

            self.all_list["train"].extend(train)
            self.all_list["unknown"].extend(test)
            self.all_list["known"].extend(random.sample(train, len(test)))

            self.classes[label_idx] = _dir.name

        self.train_size = len(self.all_list["train"])
        self.unknown_size = len(self.all_list["unknown"])
        self.known_size = len(self.all_list["known"])

        self.all_size = self.train_size + self.unknown_size

    def _write_config(self) -> None:
        r"""Writing configs."""

        _dir = Path(self.config_path)
        _dir.mkdir(parents=True, exist_ok=True)

        def _inner_execute(add_path: str, target: str):
            path = _dir.joinpath(add_path)
            file_ = ul.LogFile(path, std_debug_ok=False, _clear=True)

            for x in self.all_list[target]:
                p = Path(x.path).resolve()  # convert to absolute path
                file_.writeline(p.as_posix())

            file_.close()

        # -- train image
        _inner_execute("train_used_images.txt", "train")
        # -- unknown image
        _inner_execute("unknown_used_images.txt", "unknown")
        # -- known image
        _inner_execute("known_used_images.txt", "known")

    def create_dataloader(
        self, batch_size: int = 64, transform: transforms = None, is_shuffle: bool = True
    ) -> Dict[str, DataLoader]:
        r"""Create DataLoader instance of `train`, `unknown`, `known` dataset.

        Args:
            batch_size (int, optional): batch size. Defaults to 64.
            transform (transforms, optional): transform. Defaults to None.
            is_shuffle (bool, optional): shuffle. Defaults to True.

        Returns:
            Dict[str, DataLoader]: DataLoader.
        """

        # create dataset
        train_ = CustomDataset(self.all_list["train"], transform)
        unknown_ = CustomDataset(self.all_list["unknown"], transform)
        known_ = CustomDataset(self.all_list["known"], transform)

        train_data = DataLoader(train_, batch_size=batch_size, shuffle=is_shuffle)
        unknown_data = DataLoader(unknown_, batch_size=1, shuffle=is_shuffle)
        known_data = DataLoader(known_, batch_size=1, shuffle=is_shuffle)

        # return train_data, unknown_data, known_data
        return {"train": train_data, "unknown": unknown_data, "known": known_data}


@dataclass
class CustomDataset(Dataset):
    r"""Custom dataset

    Args:
        # dataset (CreateDataset): dataset config
        target (str, optional): target dataset. Defaults to 'train'.
        transform (transforms, optional): transform of tensor. Defaults to None.
    """
    target_list: List[Data]
    transform: transforms = None

    def __post_init__(self):
        self.list_size = len(self.target_list)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str]:
        r"""Returns image data, label, path

        Args:
            idx (int): index of list.

        Returns:
            tuple: image data, label, path
        """
        x = self.target_list[idx]
        path, label, name = x.items()

        img = Image.open(path)
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        return img, label, path

    def __len__(self):
        r"""Returns length.

        Returns:
            int: length.
        """
        return self.list_size


@dataclass
class Model:
    def __init__(
        self, classes: Dict[int, str], loader: Any, tms: _tms.TomlSettings, **options,
    ) -> None:
        r"""
        **options
            log (ul.LogFile): Defaults to ul.LogFile().
            rate (ul.LogFile): Defaults to ul.LogFile().
        """

        self.train_loader: DataLoader = loader.pop("train", None)
        self.unknown_loader: DataLoader = loader.pop("unknown", None)
        self.known_loader: DataLoader = loader.pop("known", None)

        _optim_t = Union[optim.Adam, radam.RAdam, optim.SGD]
        self.net: cnn.Net
        self.optimizer: _optim_t
        self.criterion: nn.CrossEntropyLoss

        self.tms = tms  # toml settings
        self.max_epoch = tms.epoch

        # gpu setting
        use_gpu = torch.cuda.is_available() and tms.use_gpu
        self.device = torch.device("cuda" if use_gpu else "cpu")

        # options
        self.log: ul.LogFile = options.pop("log", ul.LogFile())
        self.rate: ul.LogFile = options.pop("rate", ul.LogFile())

        self.current_epoch: int

        self.classes: Dict[int, str] = classes
        self.classify_size: int = len(classes)

        # tensorboard
        self.writer = tbx.SummaryWriter()

        # Grad-CAM setting
        self.egc = ExecuteGradCAM(
            list(self.classes.values()), tms.input_size, tms.grad_cam_layer, is_grad_cam=True
        )

        # -----

        # re-training
        if tms.is_load_model:
            self._load(tms.load_pth_path)
            self.max_epoch -= self.current_epoch  # 0 ~ (max_epoch-current_epoch) times.
        else:
            self._build()  # build model

        self._write_classes()  # save classes

        # for making false path
        base = Path(self.tms.false_path, self.tms.filename_base)
        self.false_paths = [base.joinpath(f"epoch{ep}") for ep in range(self.max_epoch)]
        ul.make_directories(*self.false_paths)

        if self.tms.pth_save_cycle != 0:
            self.pth_save_path = f"{self.tms.pth_save_path}/{self.tms.filename_base}"
            ul.make_directories(self.pth_save_path)

    def train(self):
        r"""Training model."""

        test_schedule = self._create_schedule(self.tms.test_cycle)
        pth_save_schedule = self._create_schedule(self.tms.pth_save_cycle)

        # for making confusion matrix
        all_label = torch.tensor([], dtype=torch.long)
        all_pred = torch.tensor([], dtype=torch.long)

        # switch to train
        self.net.train()

        self.log.writeline("# Start training.", False)
        plot_point = 0  # for tensorboard

        # [optimizer, epoch]
        save_option = [True, True] if self.tms.is_available_re_training else [False, False]

        # loop epoch
        for epoch in range(self.max_epoch):
            self.current_epoch = epoch + 1

            total_loss = 0  # total loss
            total_acc = 0  # total accuracy

            self.log.writeline(f"----- Epoch: {epoch + 1} -----", debug_ok=True)

            subdivision = self.tms.subdivision

            # batch in one epoch (outer tqdm)
            pbar = tqdm(
                self.train_loader,
                total=len(self.train_loader),  # subdivision
                ncols=100,
                bar_format="{l_bar}{bar:30}{r_bar}",
            )
            pbar.set_description("TRAIN")

            # batch process
            for batch_idx, items in enumerate(pbar):
                imgs: Tensor
                labels: Tensor
                paths: tuple  # if type is not [int, float...], not tensor but tuple.

                imgs, labels, paths = items

                self.optimizer.zero_grad()  # init gradient

                batch_size = len(imgs)  # batch size
                batch_result = torch.tensor([])  # all result of one batch
                batch_loss = 0  # total loss of one batch

                # generate arithmetic progression of mini batch
                sep = np.linspace(0, batch_size, subdivision + 1, dtype=np.int)

                # make grid of using train images in batch.
                # make_grid_and_plot(imgs)
                # plt.pause(0.01)

                # mini batch process
                for sd in range(subdivision):
                    n, m = sep[sd], sep[sd + 1]  # cutout data (N ~ M)
                    mb_imgs = imgs[n:m].to(self.device)
                    mb_labels = labels[n:m].to(self.device)

                    mb_result = self.net(mb_imgs)  # data into model
                    loss = self.criterion(mb_result, mb_labels)  # calculate loss
                    loss.backward()  # calculate gradient (back propagation)

                    # concatenate result
                    batch_result = torch.cat((batch_result, mb_result.cpu()), dim=0)

                    batch_loss += float(loss.item())  # add loss value
                # end of this mini batch

                self.optimizer.step()  # update parameters
                loss_val = batch_loss / subdivision  # calc avg loss value

                # tensorboard log
                self.writer.add_scalar("data/loss", loss_val, plot_point)
                plot_point += 1

                # label
                predicted = torch.max(batch_result.data, 1)[1].cpu()  # predict
                labels = labels.cpu()
                all_pred = torch.cat((all_pred, predicted), dim=0)
                all_label = torch.cat((all_label, labels), dim=0)

                predicted = predicted.numpy()  # predict
                label_ans = labels.numpy()  # correct answer

                # cls_bool = [label_ans[i] for i, x in enumerate(pred_bool) if not x]

                pred_bool = label_ans == predicted  # matching
                # index of mistake prediction
                false_step = [idx for idx, x in enumerate(pred_bool) if not x]

                # save image of mistake prediction
                for idx in false_step:
                    fp = self.false_paths[epoch]
                    name = Path(str(paths[idx])).name

                    img_path = Path(fp, f"batch_{batch_idx}-{name}")
                    img_path.parent.mkdir(parents=True, exist_ok=True)

                    save_image(imgs[idx], str(img_path))  # save

                # count of matched label
                acc_cnt = pred_bool.sum()

                # calc total
                total_acc += acc_cnt
                total_loss += loss_val * batch_size

                acc = acc_cnt / batch_size  # accuracy

                # for tqdm message
                pbar.set_postfix(
                    ordered_dict=OrderedDict(loss=f"{loss_val:<.6f}", acc=f"{acc:<.6f}")
                )

                # for log
                ss = f"loss: {loss_val:<.6f} / acc: {acc:<.6f}"
                ss += f"\n  -> ans   : {label_ans}"
                ss += f"\n  -> result: {predicted}"

                self.log.writeline(ss, debug_ok=False)

                # break
                # end of this batch

            # add confusion matrix to tensorboard
            cm = calc_confusion_matrix(all_label, all_pred, len(self.classes))
            fig = plot_confusion_matrix(cm, list(self.classes.values()))
            add_to_tensorboard(self.writer, fig, "confusion matrix", epoch)

            # calclate total loss / accuracy
            size = len(self.train_loader.dataset)
            total_loss = total_loss / size
            total_acc = total_acc / size

            # for log
            self.log.writeline("\n---------------")
            self.log.writeline(f"Total loss: {total_loss}")
            self.log.writeline(f"Total acc: {total_acc}")
            self.log.writeline("---------------\n")

            # for tqdm
            print(f"  Total loss: {total_loss}")
            print(f"  Total acc: {total_acc}\n")

            # for tensorboard
            self.writer.add_scalar("data/total_acc", total_acc, epoch)
            self.writer.add_scalar("data/total_loss", total_loss, epoch)

            # exec test cycle
            if test_schedule[epoch]:
                self.test()

            # save pth cycle
            if pth_save_schedule[epoch]:
                save_path = ul.create_file_path(
                    self.pth_save_path, "", head=f"epoch{epoch + 1}", ext="pth"
                )

                progress = ul.ProgressLog(f"Saving model to '{save_path}'")
                self.save(save_path, *save_option)  # save
                progress.complete()

                # log
                self.log.writeline(f"# Saved model to '{save_path}'")

            # break
            # end of this epoch

        # export as json
        # self.writer.export_scalars_to_json(f'{self.tms.config_path}/all_scalars.json')
        self.writer.close()

        # end of all epoch

    def test(self):
        r"""Testing model."""
        # log
        self.log.writeline("# Start testing.\n", False)

        # ss = ul.set_align_center(str(self.current_epoch)) + ' | '
        # self.rate.write(ss)
        self.rate.write(self.current_epoch)

        def _inner_execute(data_loader: DataLoader, target: str = "unknown"):
            """ Execute unknown or known dataset.

            Args:
                data_loader (DataLoader): dataloader
                tqdm_desc (str, optional): description for tqdm. Defaults to ''.
            """

            # disable gradient
            with torch.no_grad():
                self.net.eval()

                total_acc = 0  # total accuracy

                # acc of each class | {'cls_name': [acc_num, all_num]}
                acc_size = dict([(_cls, [0, 0]) for _cls in self.classes.keys()])

                # test
                pbar = tqdm(
                    data_loader,
                    total=len(data_loader.dataset),
                    ncols=100,
                    bar_format="{l_bar}{bar:30}{r_bar}",
                )
                pbar.set_description("TEST[{}]".format(target.center(7)))

                for batch_idx, items in enumerate(pbar):
                    img_data: Tensor
                    label: Tensor
                    path: tuple

                    # img_data, label, path
                    img_data, label, path = items

                    # only batch size is 1.
                    if len(path) == 1:
                        path = path[0]

                    img_data = img_data.to(self.device)
                    label = label.to(self.device)

                    out = self.net(img_data).to(self.device)  # data into model
                    predicted = torch.max(out.data, 1)[1].cpu().numpy()[0]  # predicted label
                    label_ans = label.cpu().numpy()[0]  # correct label

                    # pred_cls = self.classes[predicted]  # predicted class name
                    # ans_cls = self.classes[label_ans]  # correct class name

                    acc_size[label_ans][1] += 1  # count of test size

                    # correct -> continue
                    if predicted == label_ans:
                        acc_size[label_ans][0] += 1  # count of acc size
                        continue

                    # mistake
                    # do not Grad-CAM -> continue
                    if not self.tms.is_grad_cam:
                        continue

                    # Grad-CAM [@torch.enable_grad()]
                    ret = self.egc.main(self.net, path)

                    # base path
                    base_dir = Path(
                        self.tms.grad_cam_path,
                        self.tms.filename_base,
                        "false",
                        target,
                        f"epoch_{self.current_epoch}",
                    )
                    base_dir.mkdir(parents=True, exist_ok=True)

                    for key, data_list in ret.items():
                        for i, img_data in enumerate(data_list):
                            # save path
                            ss = f"{batch_idx}_{self.classes[i]}_{key}"
                            ss += f"_pred[{predicted}]_correct[{label_ans}].png"
                            _path = base_dir.joinpath(ss)

                            cv2.imwrite(str(_path), img_data)  # save

                            # for debug
                            # plt.imshow(img_data)
                            # plt.pause(0.1)
                    # end of this batch

                self.log.writeline()

                # calc each accuracy
                for k, _cls in self.classes.items():
                    acc_num, all_num = acc_size[k]

                    acc = acc_num / all_num
                    total_acc += acc

                    ss = "%-12s -> " % f"[{_cls}]"
                    ss += f"acc: {acc:<.4f} ({acc_num} / {all_num} images.)"
                    self.log.writeline(ss)
                    print(f"  {ss}")

                    # rate log
                    # ss = ul.set_align_ljust(str(acc), align=15)
                    # self.rate.write(ss)
                    self.rate.write(f", {acc}")

                    # end of calculate accuracy of each class and total

                self.log.writeline()

                # total accuracy
                total_acc /= len(self.classes)
                self.log.writeline(f"Total acc: {total_acc}\n")
                self.rate.write(f", {total_acc}")

                # for tqdm
                print(f"  Total acc: {total_acc}\n")

        # unknown test
        _inner_execute(self.unknown_loader, "unknown")

        # knwon test
        if self.known_loader is not None:
            _inner_execute(self.known_loader, "known")
        else:
            # spacing
            # self.rate.write(' ' * 15)
            self.rate.write(", -" * len(self.classes))

        self.rate.writeline()

    def _build(self):
        r"""Building model.

        using ...
            optimizer: `RAdam`
            criterion: `CrossEntropyLoss`
        """
        # network
        params = dict(
            input_size=self.tms.input_size,
            classify_size=self.classify_size,
            in_channels=self.tms.channels,
        )
        self.net = cnn.Net(**params)

        options = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        # self.optimizer = optim.Adam(self.net.parameters(), **options)  # Adam
        self.optimizer = radam.RAdam(self.net.parameters(), **options)  # RAdam
        # self.optimizer = optim.RAdam(self.net.parameters(), **options)  # `optim.RAdam` is unimplemented (2020/03/05).

        # self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2)

        self.criterion = nn.CrossEntropyLoss()

        self.net.zero_grad()  # init all gradient
        self.net.to(self.device)  # switch to GPU / CPU

    def _load(self, path: str) -> None:
        r"""Load model.

        Args:
            path (str): pth path.
        """
        # not exists -> raise
        ul.raise_when_FileNotFound(path)

        # load checkpoint
        checkpoint = torch.load(path)

        # classes, net
        self.classes = checkpoint["classes"]
        self.net.load_state_dict(checkpoint["model_state_dict"])

        # key is exists, load value and it assign to variable.
        # key is not exists, initialize value and it assign to variable.
        self.optimizer = checkpoint.pop("optimizer_state_dict", optim.Adam(self.net.parameters()))
        self.current_epoch = checkpoint("epoch", 0)

        self.criterion = nn.CrossEntropyLoss()

    def save(self, path: str, optimizer: bool = False, epoch: bool = False) -> None:
        r"""Save Model.

        Args:
            path (str): save path.
            optimizer (bool): save `optimizer state_dict`. Defaults to False.
            epoch (bool): save `current epoch`. Defaults to Flase.
        """
        save_config = {
            "classes": self.classes,
            "model_state_dict": self.net.state_dict(),
        }

        # options (for re-training)
        if optimizer:
            save_config["optimizer_state_dict"] = self.optimizer.state_dict()
        if epoch:
            save_config["epoch"] = self.current_epoch

        torch.save(save_config, path)

    def _write_classes(self) -> None:
        r"""Writing classes of dataset."""
        path = Path(self.tms.config_path, "classes.txt")
        file_ = ul.LogFile(path, std_debug_ok=False, _clear=True)

        path.parent.mkdir(parents=True, exist_ok=True)

        for k, _cls in self.classes.items():
            file_.writeline(f"{k}:{_cls}")
        file_.close()

    def _create_schedule(self, cycle: int) -> List[bool]:
        r"""Returns exec schedule of cycle.

        Args:
            cycle (int): exec cycle.

        Returns:
            List[bool]: exec schedule.
        """
        schedule = [False] * self.max_epoch

        if cycle == 0:
            return schedule

        # range(N - 1) -> last epoch is False.
        for i in range(self.max_epoch - 1):
            if (i + 1) % cycle == 0:
                schedule[i] = True
        return schedule


def add_to_tensorboard(
    writer: tbx.SummaryWriter, fig: plt.figure, title: str, step: int = 0
) -> None:
    r"""Add plot image to TensorBoard.

    Args:
        writer (tbx.SummaryWriter): tensorboard writer.
        fig (plt.figure): plot figure.
        title (str): title of plotted figure.
        step (int, optional): step. Defaults to 0.
    """
    fig.canvas.draw()
    img = fig.canvas.renderer._renderer
    img_ar = np.array(img).transpose(2, 0, 1)

    writer.add_image(title, img_ar)
    plt.close()  # clear plot


def plot_confusion_matrix(
    cm: Union[Tensor, np.ndarray],
    classes: List[str],
    normalize: bool = False,
    title: str = "Confusion matrix",
    cmap: plt.cm = plt.cm.Greens,
) -> plt.figure:
    r"""Plot confusion matrix.

    Args:
        cm (Union[Tensor, np.ndarray]): array of confusion matrix.
        classes (List[str]): class list.
        normalize (bool, optional): normalize. Defaults to False.
        title (str, optional): plot title. Defaults to 'Confusion matrix'.
        cmap (plt.cm, optional): using color map. Defaults to plt.cm.Greens.

    Returns:
        plt.figure: plotted figure.
    """
    # Tensor to np.ndarray
    _cm: np.ndarray = cm if not isinstance(cm, Tensor) else cm.cpu().numpy()

    if normalize:
        _cm = _cm.astype("float") / _cm.sum(axis=1)[:, np.newaxis]

    # change font size
    plt.rcParams["font.size"] = 18

    fig, axes = plt.subplots(figsize=(10, 10))

    # ticklabels
    tick_marks = np.arange(len(classes))

    plt.setp(axes, xticks=tick_marks, xticklabels=classes)
    plt.setp(axes, yticks=tick_marks, yticklabels=classes)
    # rotate xticklabels
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # title
    plt.suptitle("Confusion Matrix")

    # label
    axes.set_ylabel("True label")
    axes.set_xlabel("Predicted label")

    # grid
    # axes.grid(which='minor', color='b', linestyle='-', linewidth=3)

    img = plt.imshow(cm, interpolation="nearest", cmap=cmap)

    # adjust color bar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(img, cax=cax)

    thresh = cm.max() / 2.0
    fmt = ".2f" if normalize else "d"

    # plot text
    for i, j in itertools.product(range(len(classes)), range(len(classes))):
        clr = "white" if cm[i, j] > thresh else "black"
        axes.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color=clr)

    plt.tight_layout()
    fig = plt.gcf()
    return fig


def calc_confusion_matrix(
    correct_labels: Union[Tensor, np.ndarray],
    predicted_labels: Union[Tensor, np.ndarray],
    class_num: int,
) -> Union[Tensor, np.ndarray]:
    r"""Calculating confusion matrix.

    Args:
        correct_labels (Union[Tensor, np.ndarray]): array of correct labels.
        predicted_labels (Union[Tensor, np.ndarray]): array of predicted labels.
        class_num (int): class num.

    Returns:
        Union[Tensor, np.ndarray]: confusion matrix.
    """
    cm = torch.zeros(class_num, class_num, dtype=torch.int64)
    stacked = torch.stack((correct_labels, predicted_labels), dim=1)

    for p in stacked:
        tl, pl = p.tolist()
        cm[tl, pl] += 1

    return cm


def calc_dataset_norm(dataset: CreateDataset, channels: int = 3):
    """Returns mean and std."""
    CHANNEL_NUM = channels

    # xp: Union[np, cupy] = cupy if torch.cuda.is_available() else np
    xp: np = np

    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = xp.zeros(CHANNEL_NUM)
    channel_sum_squared = xp.zeros(CHANNEL_NUM)

    for key, data_list in dataset.all_list.items():
        if key == "known":
            continue

        # for i, data in enumerate(data_list):
        for data in data_list:
            # print(f'\r{i}: {data.path}', end='')
            # img = cv2.imread(data.path)
            # img = cv2.resize(img, (60, 60))
            img_pil = Image.open(data.path).resize((60, 60))

            img = xp.asarray(img_pil)
            img = img / 255
            pixel_num += img.size / CHANNEL_NUM
            channel_sum += xp.sum(img, axis=(0, 1))
            channel_sum_squared += xp.sum(xp.square(img), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = xp.sqrt(channel_sum_squared / pixel_num - xp.square(bgr_mean))

    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    return tuple(rgb_mean), tuple(rgb_std)


def make_grid_and_plot(imgs: Tensor) -> None:
    imgs = torchvision.utils.make_grid(imgs)
    imgs = imgs / 2 + 0.5
    np_imgs = imgs.cpu().numpy()
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)))
