import itertools
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorboardX as tbx
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

# my packages
import cnn
import toml_settings as _tms
import utils as ul
from grad_cam import ExecuteGradCAM


@dataclass
class Data:
    r""" Nesessary datas of create dataset. """

    path: str  # image path
    label: int  # label of class
    name: str  # class name

    def items(self):
        r""" Returns items of Data class.

        Returns:
            tuple: Items of class.
        """

        return self.path, self.label, self.name
    # end of [function] items
# end of [class] Data


@dataclass(init=False)
class CreateDataset(Dataset):
    r""" Creating dataset. """

    def __init__(
            self,
            path: str,
            extensions: list,
            test_size: Union[float, int],
            config_path: str = 'config'):
        r"""
        Args:
            path (str): path of image directory.
            extensions (list): supported extensions.
            test_size (Union[float, int]): test image size.
            config_path (str, optional): export path of config. Defaults to 'config'.
        """

        self.path = path
        self.extensions = extensions
        self.test_size = test_size
        self.config_path = config_path

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

        # train_list / test_list
        self._get_all_datas()

        # write config of model
        self._write_config()
    # end of [function] __init__

    def _get_all_datas(self):
        """ Get all datas from each directory. """

        # init
        self.all_list = {'train': [], 'unknown': [], 'known': []}
        self.classes = {}

        path = Path(self.path)

        # directories in [image_path]
        dirs = [d for d in path.glob('*') if d.is_dir()]

        # all extensions / all sub directories
        for idx, _dir in enumerate(dirs):
            xs = []

            for ext in self.extensions:
                tmp = [Data(x.as_posix(), idx, _dir.name)
                       for x in _dir.glob(f'*.{ext}') if x.is_file()]
                xs.extend(tmp)

            # split dataset
            train, test = train_test_split(
                xs, test_size=self.test_size, shuffle=True)

            self.all_list['train'].extend(train)
            self.all_list['unknown'].extend(test)
            self.all_list['known'].extend(random.sample(train, len(test)))

            self.classes[idx] = _dir.name

        self.train_size = len(self.all_list['train'])
        self.unknown_size = len(self.all_list['unknown'])
        self.known_size = len(self.all_list['known'])

        self.all_size = self.train_size + self.unknown_size
    # end of [function] __get_all_datas

    def _write_config(self):
        """ Writing configs. """

        _dir = Path(self.config_path)
        _dir.mkdir(parents=True, exist_ok=True)

        def _inner_execute(add_path: str, target: str):
            path = _dir.joinpath(add_path)
            file_ = ul.LogFile(path, std_debug_ok=False)

            for x in self.all_list[target]:
                p = Path(x.path).resolve()  # convert to absolute path
                file_.writeline(p.as_posix())

            file_.close()

        # -- train image
        _inner_execute('train_used_images.txt', 'train')
        # -- unknown image
        _inner_execute('unknown_used_images.txt', 'unknown')
        # -- known image
        _inner_execute('known_used_images.txt', 'known')
    # end of [function] __write_config

    def create_dataloader(
            self,
            batch_size: int = 64,
            transform: transforms = None,
            is_shuffle: bool = True):

        # create dataset
        train_ = CustomDataset(self.all_list['train'], transform)
        unknown_ = CustomDataset(self.all_list['unknown'], transform)
        known_ = CustomDataset(self.all_list['known'], transform)

        train_data = DataLoader(train_, batch_size=batch_size, shuffle=is_shuffle)
        unknown_data = DataLoader(unknown_, batch_size=1, shuffle=is_shuffle)
        known_data = DataLoader(known_, batch_size=1, shuffle=is_shuffle)

        # return train_data, unknown_data, known_data
        return {
            'train': train_data,
            'unknown': unknown_data,
            'known': known_data
        }
    # end of [function] create_dataloader
# end of [class] CreateDataset


@dataclass(init=False)
class CustomDataset(Dataset):
    """ Custom dataset. """

    def __init__(
            self,
            target_list: list,
            transform: transforms = None):
        """
        Args:
            # dataset (CreateDataset): dataset config
            target (str, optional): target dataset. Defaults to 'train'.
            transform (transforms, optional): transform of tensor. Defaults to None.
        """

        self.transform = transform
        self.target_list = target_list
        self.list_size = len(self.target_list)
    # end of [function] __init__

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str]:
        """ Returns image data, label, path

        Args:
            idx (int): index of list.

        Returns:
            tuple: image data, label, path
        """
        x = self.target_list[idx]
        path, label, name = x.items()

        img = Image.open(path)
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label, path
    # end of [function] __getitem__

    def __len__(self):
        """ Returns length.

        Returns:
            int: length.
        """
        return self.list_size
    # end of [function] __len__
# end of [class] CustomDataset


@dataclass(init=False)
class Model:
    """ Basic parameters for model. """
    def __init__(
            self,
            toml_settings: Optional[_tms.TomlSettings] = None,
            classes: Optional[Dict[int, str]] = None,
            use_gpu: bool = True,
            **options):
        """[summary]

        Args:
            toml_settings (Optional[_tms.TomlSettings], optional): [description]. Defaults to None.
            classes (Optional[Dict[int, str]], optional): [description]. Defaults to None.
            use_gpu (bool, optional): [description]. Defaults to True.

        **options
            log (ul.LogFile): Defaults to None.
            rate (ul.LogFile): Defaults to None.
        """

        # **options
        log = options.pop('log', None)
        rate = options.pop('rate', None)

        # network configure
        self.net: cnn.Net  # network
        self.optimizer: Union[optim.Adam, optim.SGD]  # optimizer
        self.criterion: nn.CrossEntropyLoss  # criterion

        self.classes: Dict[int, str]  # class
        self.input_size = toml_settings.input_size  # image size when input to network

        # gpu setting
        self.use_gpu = torch.cuda.is_available() and use_gpu
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        if toml_settings is None:
            toml_settings = _tms.factory()
        self.tms = toml_settings

        # for tensorboard
        self.writer = tbx.SummaryWriter()

        # assign or load from classes.txt
        cls_txt = f'{toml_settings.config_path}/classes.txt'
        self.classes = classes if classes is not None \
            else ul.load_classes(cls_txt).copy()

        # create instance if log is None
        self.log = log if log is not None else ul.LogFile(None)

        # create instance if rate is None
        self.rate = rate if rate is not None else ul.LogFile(None)

        # build model
        self._build_model()

        # save classes
        self._write_classes()
    # end of [function] __init__

    def _build_model(self):
        self.net = cnn.Net(input_size=self.input_size)  # network

        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        self.criterion = nn.CrossEntropyLoss()

        self.net.zero_grad()  # init all gradient
        self.net.to(self.device)  # switch to GPU / CPU
    # end of [function] _build_model

    def inherit_params(self, model: 'Model'):
        """ Inherit from [TrainModel], [TestModel].

        Args:
            model (Model): model parameters.
        """

        # self.XX = YY
        for k in model.__dict__.keys():
            expression = f'self.{k} = model.{k}'
            exec(expression)
    # end of [function] inherit_params

    def save_model(self, path: str):
        """ Save Model.

        Args:
            path (str): save path.
            epoch (Optional[int], optional): epoch num. Defaults to None.
        """

        torch.save({
            'classes': self.classes,
            'model_state_dict': self.net.state_dict(),

            # 'epoch': epoch,
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            # 'criterion': type(self.criterion).__name__
        }, path)
    # end of [function] save_model

    def _write_classes(self):
        """ Writing classes of dataset. """

        _dir = Path(self.tms.config_path)
        _dir.mkdir(parents=True, exist_ok=True)

        # -- write classes
        path = _dir.joinpath('classes.txt')
        with open(path, 'w') as f:
            for k, _cls in self.classes.items():
                f.write(f'{k}:{_cls}\n')
    # end of [function] __write_classes
# end of [class] Model


@dataclass(init=False)
class TestModel(Model):
    def __init__(
            self,
            model: Model,
            unknown_loader: DataLoader,
            known_loader: Optional[DataLoader] = None):
        """
        Args:
            model (Model): model parameters.
            unknown_data (DataLoader): unknown dataloader.
            known_data (Optional[DataLoader], optional): known dataloader. Defaults to None.
        """

        self.inherit_params(model)  # inherit

        # dataloader
        self.unknown_data = unknown_loader
        self.known_data = known_loader

        # Grad-CAM
        self.egc = ExecuteGradCam(
            list(self.classes.values()),
            self.input_size,
            self.tms.grad_cam_layer,
            self.tms.grad_cam_path)
    # end of [function] __init__

    def test(self, epoch: Optional[int] = None):
        self.log.writeline('# Start testing.\n', False)

        def _inner_execute(data_loader: DataLoader, target: str = 'unknown'):
            """ Execute unknown or known dataset.

            Args:
                data_loader (DataLoader): dataloader
                tqdm_desc (str, optional): description for tqdm. Defaults to ''.
            """

            with torch.no_grad():
                self.net.eval()

                total_acc = 0  # total accuracy

                # acc of each class | {'cls_name': [acc_num, all_num]}
                acc_size = dict([(_cls, [0, 0]) for _cls in self.classes.keys()])

                # test
                pbar = tqdm(data_loader, total=len(data_loader),
                            ncols=100, bar_format='{l_bar}{bar:30}{r_bar}')
                pbar.set_description('TEST[{}]'.format(target.center(7)))

                # for data, label, name in self.test_data:
                for batch_idx, items in enumerate(pbar):
                    # img, label, path
                    img, label, path = items

                    img = img.to(self.device)
                    label = label.to(self.device)

                    out = self.net(img).to(self.device)  # data into model
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

                    # Grad-CAM
                    ret = self.egc.main(self.net, path)

                    # base path
                    exp_path_base = Path(self.tms.grad_cam_path, self.tms.filename_base,
                                         'false', target, f'epoch_{epoch + 1}')
                    exp_path_base.mkdir(parents=True, exist_ok=True)

                    for key, _list in ret.items():
                        for i, img in enumerate(_list):
                            # save path
                            ss = f'{batch_idx}_{self.classes[i]}_{key}'
                            ss += f'_pred[{predicted}]_correct[{label_ans}].png'
                            _path = exp_path_base.joinpath(ss)

                            cv2.imwrite(str(_path), img)  # save

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

                    ss = '%-12s -> ' % f'[{_cls}]'
                    ss += f'acc: {acc:<.4f} ({acc_num} / {all_num} images.)'
                    self.log.writeline(ss, debug_ok=False)
                    print(f'  {ss}')

                    # end of calculate accuracy of each class and total

                self.log.writeline()

                # total accuracy
                total_acc /= len(self.classes)
                self.log.writeline(f'Total acc: {total_acc}\n', debug_ok=False)

                # for tqdm
                print(f'  Total acc: {total_acc}\n')
        # end of [function] __execute

        # unknown test
        _inner_execute(self.unknown_data, 'unknown')

        # knwon test
        if self.known_data is not None:
            _inner_execute(self.known_data, 'known')
    # end of [function] test
# end of [class] TestModel


@dataclass(init=False)
class TrainModel(Model):
    def __init__(
            self,
            model: Model,
            epoch: int,
            train_loader: DataLoader,
            test_model: TestModel) -> None:
        """
        Args:
            model (Model): model parameters.
            epoch (int): max epoch.
            train_data (DataLoader): train dataloader.
            test_model (TestModel): testing model.
        """

        # parameters
        self.inherit_params(model)  # inherit by model variable

        self.max_epoch = epoch
        self.train_loader = train_loader
        self.test_model = test_model  # TestModel

        # schedule by cycle
        self.test_schedule = self._create_schedule(self.tms.test_cycle)
        self.pth_save_schedule = self._create_schedule(self.tms.pth_save_cycle)

        # for making confusion matrix
        self.all_label = torch.tensor([], dtype=torch.long)
        self.all_pred = torch.tensor([], dtype=torch.long)

        # for making false path
        base = Path(self.tms.false_path, self.tms.filename_base)
        self.false_paths = [base.joinpath(f'epoch{ep}') for ep in range(self.max_epoch)]
        ul.make_directories(*self.false_paths)

        if self.tms.pth_save_cycle != 0:
            self.pth_save_path = f'{self.tms.pth_save_path}/{self.tms.filename_base}'
            ul.make_directories(self.pth_save_path)

    # end [function] __init__

    def train(self):
        """ Training model. """

        # switch to train
        self.net.train()

        self.log.writeline('# Start training.', False)
        plot_point = 0  # for tensorboard

        # loop epoch
        for ep in range(self.max_epoch):
            total_loss = 0  # total loss
            total_acc = 0  # total accuracy

            self.log.writeline(f'----- Epoch: {ep + 1} -----')

            subdivision = self.tms.subdivision

            # batch in one epoch (outer tqdm)
            outer_pbar = tqdm(self.train_loader, total=len(self.train_loader),
                              ncols=100, bar_format='{l_bar}{bar:30}{r_bar}')
            outer_pbar.set_description('TRAIN')

            # batch process
            for batch_idx, items in enumerate(outer_pbar):
                imgs: torch.Tensor
                labels: torch.Tensor
                paths: torch.Tensor

                imgs, labels, paths = items

                self.optimizer.zero_grad()  # init gradient

                batch_size = len(imgs)  # batch size
                batch_result = torch.tensor([])  # all result of one batch
                batch_loss = 0  # total loss of one batch

                # generate arithmetic progression of mini batch
                sep = np.linspace(0, batch_size, subdivision + 1, dtype=np.int)

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
                self.writer.add_scalar('data/loss', loss_val, plot_point)
                plot_point += 1

                # label
                predicted = torch.max(batch_result.data, 1)[1].cpu()  # predict
                labels = labels.cpu()
                self.all_pred = torch.cat((self.all_pred, predicted), dim=0)
                self.all_label = torch.cat((self.all_label, labels), dim=0)

                predicted = predicted.numpy()  # predict
                label_ans = labels.numpy()  # correct answer

                # cls_bool = [label_ans[i] for i, x in enumerate(pred_bool) if not x]

                pred_bool = (label_ans == predicted)  # matching
                # index of mistake prediction
                false_step = [idx for idx, x in enumerate(pred_bool) if not x]

                # save image of mistake prediction
                for idx in false_step:
                    fp = self.false_paths[ep]
                    name = Path(str(paths[idx])).name

                    img_path = Path(fp, f'batch_{batch_idx}-{name}')
                    img_path.parent.mkdir(parents=True, exist_ok=True)

                    save_image(imgs[idx], str(img_path))  # save

                # count of matched label
                acc_cnt = pred_bool.sum()

                # calc total
                total_acc += acc_cnt
                total_loss += loss_val * batch_size

                acc = acc_cnt / batch_size  # accuracy

                # for tqdm message
                outer_pbar.set_postfix(
                    ordered_dict=OrderedDict(loss=f'{loss_val:<.6f}', acc=f'{acc:<.6f}'))

                # for log
                ss = f'loss: {loss_val:<.6f} / acc: {acc:<.6f}'
                ss += f'\n  -> ans   : {label_ans}'
                ss += f'\n  -> result: {predicted}'

                self.log.writeline(ss, debug_ok=False)

                # break
                # end of this batch

            # add confusion matrix to tensorboard
            cm = calc_confusion_matrix(self.all_label, self.all_pred, len(self.classes))
            fig = plot_confusion_matrix(cm, list(self.classes.values()))
            add_to_tensorboard(self.writer, fig, 'confusion matrix', ep)

            # calclate total loss / accuracy
            size = len(self.train_loader.dataset)
            total_loss = total_loss / size
            total_acc = total_acc / size

            # for log
            self.log.writeline('\n---------------', debug_ok=False)
            self.log.writeline(f'Total loss: {total_loss}', debug_ok=False)
            self.log.writeline(f'Total acc: {total_acc}', debug_ok=False)
            self.log.writeline('---------------\n', debug_ok=False)

            # for tqdm
            print(f'  Total loss: {total_loss}')
            print(f'  Total acc: {total_acc}\n')

            # for tensorboard
            self.writer.add_scalar('data/total_acc', total_acc, ep)
            self.writer.add_scalar('data/total_loss', total_loss, ep)

            # exec test cycle
            if self.test_schedule[ep]:
                self.test_model.test(epoch=ep)

            # save pth cycle
            if self.pth_save_schedule[ep]:
                save_path = ul.create_file_path(
                    self.pth_save_path, '', head=f'epoch{ep + 1}', ext='pth')

                progress = ul.ProgressLog(f'Saving model to \'{save_path}\'')
                self.save_model(save_path)  # save
                progress.complete()

                # log
                self.log.writeline(f'# Saved model to \'{save_path}\'', debug_ok=False)

            # break
            # end of this epoch

        # export as json
        # self.writer.export_scalars_to_json(f'{self.tms.config_path}/all_scalars.json')
        self.writer.close()

        # end of all epoch
    # end of [function] train

    def _create_schedule(self, cycle: int) -> List[bool]:
        """ Returns exec schedule of cycle.

        Args:
            cycle (int): exec cycle.

        Returns:
            List[bool]: exec schedule.
        """

        _list = [False] * self.max_epoch

        if cycle == 0:
            return _list

        # range(N - 1) -> last epoch is False.
        for i in range(self.max_epoch - 1):
            if (i + 1) % cycle == 0:
                _list[i] = True
        return _list
    # end of [function] __create_schedule
# end of [class] TrainModel


def add_to_tensorboard(
        writer: tbx.SummaryWriter,
        fig: plt.figure,
        title: str,
        step: int = 0):

    fig.canvas.draw()
    img = fig.canvas.renderer._renderer
    img_ar = np.array(img).transpose(2, 0, 1)

    writer.add_image(title, img_ar)
    plt.close()
# end of [function] add_to_tensorboard


def plot_confusion_matrix(
        cm: Union[torch.Tensor, np.ndarray],
        classes: List[str],
        normalize: bool = False,
        title: str = 'Confusion matrix',
        cmap: plt.cm = plt.cm.Greens):

    # torch.Tensor to np.ndarray
    _cm: np.ndarray = cm if not isinstance(cm, torch.Tensor) \
        else cm.cpu().numpy()

    if normalize:
        _cm = _cm.astype('float') / _cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # change font size
    plt.rcParams["font.size"] = 18

    fig, axes = plt.subplots(figsize=(10, 10))

    # ticklabels
    tick_marks = np.arange(len(classes))

    plt.setp(axes, xticks=tick_marks, xticklabels=classes)
    plt.setp(axes, yticks=tick_marks, yticklabels=classes)
    # rotate xticklabels
    plt.setp(axes.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # title
    plt.suptitle('Confusion Matrix')

    # label
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')

    # grid
    # axes.grid(which='minor', color='b', linestyle='-', linewidth=3)

    img = plt.imshow(cm, interpolation='nearest', cmap=cmap)

    # adjust color bar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(img, cax=cax)

    thresh = cm.max() / 2.
    fmt = '.2f' if normalize else 'd'

    # plot text
    for i, j in itertools.product(range(len(classes)), range(len(classes))):
        clr = 'white' if cm[i, j] > thresh else 'black'
        axes.text(j, i, format(cm[i, j], fmt), ha='center', va='center', color=clr)

    plt.tight_layout()
    fig = plt.gcf()
    return fig
# end of [function] plot_confusion_matrix


def calc_confusion_matrix(
        correct_labels: Union[torch.Tensor, np.ndarray],
        predicted_labels: Union[torch.Tensor, np.ndarray],
        class_num: int):

    cm = torch.zeros(class_num, class_num, dtype=torch.int64)
    stacked = torch.stack((correct_labels, predicted_labels), dim=1)

    for p in stacked:
        tl, pl = p.tolist()
        cm[tl, pl] += 1

    return cm
# end of [function] calc_confusion_matrix
