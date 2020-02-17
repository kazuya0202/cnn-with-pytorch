import itertools
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

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
from grad_cam import ExecuteGradCam


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
class CreateDataset:
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
        """

        self.path: str = path
        self.extensions: List[str] = extensions
        self.test_size: Union[float, int] = test_size
        self.config_path: str = config_path

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
        self.__get_all_datas()

        # write config of model
        self.__write_config()
    # end of [function] __init__

    def __get_all_datas(self):
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

    def __write_config(self):
        """ Writing configs. """

        _dir = Path(self.config_path)
        _dir.mkdir(parents=True, exist_ok=True)

        def __inner_execute(path: Path, target: str, head: str = ''):
            with path.open('w') as f:
                f.write(head)

                for x in self.all_list[target]:
                    p = Path(x.path).resolve()  # convert to absolute path
                    f.write(p.as_posix() + '\n')

        # # -- train image path
        path = _dir.joinpath('train_used_images.txt')
        __inner_execute(path, 'train')
        # __execute(path, 'train', '--- Image used for training. ---\n')

        # # -- unknown image path
        path = _dir.joinpath('unknown_used_images.txt')
        __inner_execute(path, 'unknown')
        # __execute(path, 'unknown', '--- Image used for unknown testing. ---\n')

        # # -- known image path
        path = _dir.joinpath('known_used_images.txt')
        __inner_execute(path, 'known')
        # __execute(path, 'known', '--- Image used for known testing. ---\n')

    # end of [function] __write_config
# end of [class] CreateDataset


@dataclass(init=False)
class CustomDataset(Dataset):
    """ Custom dataset. """

    def __init__(
            self,
            dataset: CreateDataset,
            target: str = 'train',
            transform: transforms = None):
        """
        Args:
            dataset (CreateDataset): dataset config
            target (str, optional): target dataset. Defaults to 'train'.
            transform (transforms, optional): transform of tensor. Defaults to None.
        """

        self.dataset = dataset
        self.transform = transform
        self.target_list = self.dataset.all_list[target]
        self.list_size = len(self.target_list)
    # end of [function] __init__

    def __getitem__(self, idx: int):
        """ Returns image data, label, class name

        Args:
            idx (int): index of list.

        Returns:
            tuple: image data, label, name
        """
        x = self.target_list[idx]
        path, label, name = x.items()

        img = Image.open(path)
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label, name, path
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
            toml_settings: _tms.TomlSettings,
            classes: Optional[dict] = None,
            use_gpu: bool = True,
            log: Optional[ul.LogFile] = None,
            rate: Optional[ul.LogFile] = None,
            load_pth_path: Optional[str] = None) -> None:
        """
        Args:
            toml_settings (_tms.TomlSettings): parameters of user's setting.
            classes (Optional[dict], optional): classes of dataset. Defaults to None.
            use_gpu (bool, optional): uisng gpu or cpu. Defaults to False.
            log (Optional[ul.LogFile], optional): logging for debug log. Defaults to None.
            rate (Optional[ul.LogFile], optional): logging for rate log. Defaults to None.
            load_pth_path (Optional[str], optional): pth(model) path. Defaults to None.

        `load_pth_path` is not None -> load pth automatically.
        `classes` is None -> load from 'config/classes.txt' if pth load is False,
                             load from pth checkpoint if pth laod is True.
        `use_gpu` is assigned False automatically if cuda is not available.
        """

        # network configure
        self.net: cnn.Net  # network
        self.optimizer: Union[optim.Adam, optim.SGD]  # optimizer
        self.criterion: nn.CrossEntropyLoss  # criterion

        self.classes: Dict[int, str]  # class
        self.input_size = toml_settings.input_size  # image size when input to network
        self.load_pth_path = load_pth_path  # pth path
        self.current_epoch: int  # for load pth

        # gpu setting
        self.use_gpu = torch.cuda.is_available() and use_gpu
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        self.tms = toml_settings

        # for tensorboard
        self.writer = tbx.SummaryWriter()

        # assign or load from classes.txt
        if load_pth_path is None:
            cls_txt = f'{toml_settings.config_path}/classes.txt'
            self.classes = classes if classes is not None \
                else ul.load_classes(cls_txt).copy()

        # create instance if log is None
        self.log = log if log is not None else ul.LogFile(None)

        # create instance if rate is None
        self.rate = rate if rate is not None else ul.LogFile(None)

        # build model by cnn.py or load model
        self.__build_model()

        # save classes
        self.__write_classes()

        # classes.txt is EOF or classes is None
        if self.classes is {}:
            print('classes is {}')
    # end of [function] __init__

    def inherit_params(self, model: 'Model'):
        """ Inherit from [TrainModel], [TestModel].

        Args:
            model (Model): model parameters.
        """
        for k in model.__dict__.keys():
            expression = f'self.{k} = model.{k}'
            exec(expression)

        # network params
        # self.net = model.net
        # self.criterion = model.criterion
        # self.optimizer = model.optimizer

        # # configs
        # self.classes = model.classes
        # self.device = model.device
        # self.log = model.log  # log
        # self.rate = model.rate  # rate
        # self.use_gpu = model.use_gpu
        # self.input_size = model.input_size
        # self.load_pth_path = model.load_pth_path
        # self.tms = model.tms

        # # for tensorboard
        # self.writer = model.writer
    # end of [function] inherit_params

    def __build_model(self):
        r""" Building model.

        * load pth
            -> network: load from pth model.
            -> optimizer: Adam algorithm.
            -> criterion: Cross Entropy Loss algorithm.

            -> classes: load from pth model.
            -> epoch: load from pth model.

        * do not load pth
            -> network: Net(cnn.py) instance and init gradient.
            -> optimizer: Adam algorithm.
            -> criterion: Cross Entropy Loss algorithm.
        """

        self.net = cnn.Net(input_size=self.input_size)  # network

        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=pow(10, -8))
        self.criterion = nn.CrossEntropyLoss()

        # load
        if self.load_pth_path is None:
            self.net.zero_grad()  # init all gradient
        else:
            # check exist
            if not Path(self.load_pth_path).exists():
                print(f'{self.load_pth_path} is not exist.')

            # load checkpoint
            checkpoint = torch.load(self.load_pth_path)

            # classes, network
            self.classes = checkpoint['classes']
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.current_epoch = checkpoint['epoch']

            # criterion = checkpoint['criterion']
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.net.to(self.device)  # switch to GPU / CPU
    # end of [function] __build_model

    def save_model(self, path: str, epoch: Optional[int] = None, _save: bool = False):
        """ Save Model.

        Args:
            path (str): save path.
            epoch (Optional[int], optional): epoch num. Defaults to None.
            _save (bool, optional): save. for debug. Defaults to False.
        """

        if not _save:  # for debug
            return

        torch.save({
            'classes': self.classes,
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),

            # 'optimizer_state_dict': self.optimizer.state_dict(),
            # 'criterion': type(self.criterion).__name__
        }, path)
    # end of [function] save_model

    def __write_classes(self):
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
            unknown_data: DataLoader,
            known_data: Optional[DataLoader] = None):
        """
        Args:
            model (Model): model parameters.
            unknown_data (DataLoader): unknown dataloader.
            known_data (Optional[DataLoader], optional): known dataloader. Defaults to None.
        """

        self.inherit_params(model)  # inherit

        # dataloader
        self.unknown_data = unknown_data
        self.known_data = known_data

        # Grad-CAM
        self.egc = ExecuteGradCam(
            list(self.classes.values()),
            self.input_size,
            self.tms.grad_cam_layer,
            self.tms.grad_cam_path)
    # end of [function] __init__

    def test(self, epoch: Optional[int] = None):
        self.net.eval()
        self.log.writeline('# Start testing.\n', False)

        def __inner_execute(data_loader: DataLoader, target: str = 'unknown'):
            """ Execute unknown or known dataset.

            Args:
                data_loader (DataLoader): dataloader
                tqdm_desc (str, optional): description for tqdm. Defaults to ''.
            """

            total_acc = 0  # total accuracy

            # acc of each class | {'cls_name': [acc_num, all_num]}
            acc_size = dict([(_cls, [0, 0]) for _cls in self.classes.keys()])

            # test
            pbar = tqdm(data_loader, total=len(data_loader),
                        ncols=100, bar_format='{l_bar}{bar:30}{r_bar}')
            pbar.set_description('TEST[{}]'.format(target.center(7)))

            # for data, label, name in self.test_data:
            for idx, items in enumerate(pbar):
                img_data, label, name, path = items

                img_data = img_data.to(self.device)
                label = label.to(self.device)

                # data into model
                out = self.net(img_data).to(self.device)

                # label
                predicted = torch.max(out.data, 1)[1].cpu().numpy()[0]  # predict
                label_ans = label.cpu().numpy()[0]  # correct answer

                # release memory
                # del out, label, img, name, path

                # pred_cls = self.classes[predicted]
                # ans_cls = self.classes[label_ans]

                acc_size[label_ans][1] += 1

                # correct
                if predicted == label_ans:
                    acc_size[label_ans][0] += 1
                    continue

                # mistake
                # grad-cam
                if not self.tms.is_grad_cam:
                    continue

                # prevent memory error
                self.net.to('cpu')

                # grad-cam
                ret = self.egc.main(self.net, path)

                exp_path_base = Path(self.tms.grad_cam_path, 'false', target, f'epoch_{epoch + 1}')
                exp_path_base.mkdir(parents=True, exist_ok=True)

                for key, _list in ret.items():
                    for i, img_data in enumerate(_list):
                        _path = exp_path_base.joinpath(
                            f'{idx}_{self.classes[i]}_{key}_pred[{predicted}]_correct[{label_ans}].png')

                        # plt.imshow(img_data)
                        # plt.pause(0.1)

                        # save
                        cv2.imwrite(str(_path), img_data)

                    # restore device
                    self.net.to(self.device)

                # label of correct answer
                # count of matched label, and it add to total_acc
            self.log.writeline()

            # calc each accuracy
            for k, _cls in self.classes.items():
                acc_num, all_num = acc_size[k]

                acc = acc_num / all_num
                total_acc += acc

                ss = f'%-12s -> acc: {acc:<.4f} ({acc_num} / {all_num} images.)' % f'[{_cls}]'
                self.log.writeline(ss, debug_ok=False)
                print(f'  {ss}')

                # end of calculate accuracy of each class and total

            self.log.writeline()

            # total accuracy
            total_acc /= len(self.classes)
            print('  ', end='')
            self.log.writeline(f'Total acc: {total_acc}\n')
        # end of [function] __execute

        __inner_execute(self.unknown_data, 'unknown')  # unknown test

        if self.known_data is not None:
            __inner_execute(self.known_data, 'known')  # knwon test

    # def test(self, epoch: Optional[int] = None):
    #     """ Testing model
    #     batchsize > 1 (difficult gradcam)

    #     Args:
    #         epoch (Optional[int], optional): current epoch. Defaults to None.
    #     """

    #     # switch to test
    #     self.net.eval()

    #     self.log.writeline('# Start testing.\n', False)

    #     acc_result = {
    #         'known': {},
    #         'unknown': {},
    #     }

    #     def __inner_execute(data_loader: DataLoader, target: str = 'unknown'):
    #         """ Execute unknown or known dataset.

    #         Args:
    #             data_loader (DataLoader): dataloader
    #             tqdm_desc (str, optional): description for tqdm. Defaults to ''.
    #         """

    #         total_acc = 0  # total accuracy

    #         _dict = dict([(_cls, [0, 0]) for _cls in self.classes.keys()])

    #         # acc of each class
    #         acc_size = _dict.copy()  # {'cls_name': [acc_num, all_num]}
    #         # all_size = _dict.copy()  # test size of each class

    #         # test
    #         pbar = tqdm(data_loader, total=len(data_loader),
    #                     ncols=100, bar_format='{l_bar}{bar:30}{r_bar}')
    #         pbar.set_description('TEST[{}]'.format(target.center(7)))

    #         # for data, label, name in self.test_data:
    #         for items in pbar:
    #             imgs, labels, names, paths = items

    #             imgs = imgs.to(self.device)
    #             labels = labels.to(self.device)

    #             # data into model
    #             out = self.net(imgs).to(self.device)

    #             # label
    #             predicted = torch.max(out.data, 1)[1].cpu().numpy()  # predict
    #             label_ans = labels.cpu().numpy()  # correct answer

    #             # release memory
    #             del out, labels, imgs, names, paths

    #             # label of correct answer
    #             # count of matched label, and it add to total_acc
    #             for k in self.classes.keys():
    #                 # match -> 1, else -> -1 / -2
    #                 x = np.where(label_ans == k, 1, -1)
    #                 y = np.where(predicted == k, 1, -2)

    #                 acc_size[k][0] += (x == y).sum()  # match -> correct
    #                 acc_size[k][1] += x.tolist().count(1)  # all size of each class
    #                 # acc_size[k] += (x == y).sum()  # match -> correct
    #                 # all_size[k] += x.tolist().count(1)  # all size of each class

    #             # break
    #             # end of this test
    #         # end of all test
    #         self.log.writeline()

    #         # calc each accuracy
    #         for k, _cls in self.classes.items():
    #             # acc_num = acc_size[k]
    #             # all_num = all_size[k]
    #             acc_num, all_num = acc_size[k]

    #             acc = acc_num / all_num
    #             total_acc += acc
    #             acc_result[target][_cls] = acc

    #             ss = f'%-12s -> acc: {acc:<.4f} ({acc_num} / {all_num} images.)' % f'[{_cls}]'
    #             self.log.writeline(ss, debug_ok=False)
    #             print(f'  {ss}')

    #             # end of calculate accuracy of each class and total

    #         self.log.writeline()

    #         # total accuracy
    #         total_acc /= len(self.classes)
    #         print('  ', end='')
    #         self.log.writeline(f'Total acc: {total_acc}\n')
    #     # end of [function] __execute

    #     __inner_execute(self.unknown_data, 'unknown')  # unknown test

    #     if self.known_data is not None:
    #         __inner_execute(self.known_data, 'known')  # knwon test

    #     # tensorboard (do not plot when last epoch.)
    #     # for target, cls_dict in acc_result.items():
    #     #     self.writer.add_scalars(f'test/acc_{target}', cls_dict, epoch)
    # # end of [function] test

# end of [class] TestMedel


@dataclass(init=False)
class TrainModel(Model):
    def __init__(
            self,
            model: Model,
            epoch: int,
            train_data: DataLoader,
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
        self.train_data = train_data
        self.test_model = test_model  # TestModel

        # schedule by cycle
        self.test_schedule = self.__create_schedule(self.tms.test_cycle)
        self.pth_save_schedule = self.__create_schedule(self.tms.pth_save_cycle)

        # for making confusion matrix
        self.all_label = torch.tensor([], dtype=torch.long)
        self.all_pred = torch.tensor([], dtype=torch.long)

        self.false_paths = [Path(self.tms.false_path, f'epoch{ep}') for ep in range(self.max_epoch)]

        # mkdir
        for path in self.false_paths:
            path.mkdir(parents=True, exist_ok=True)

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

            # batch in one epoch
            outer_pbar = tqdm(self.train_data, total=len(self.train_data),
                              ncols=100, bar_format='{l_bar}{bar:30}{r_bar}')

            # batch process
            for batch_idx, items in enumerate(outer_pbar):
                imgs, labels, names, paths = items
                self.optimizer.zero_grad()  # init gradient

                split = np.linspace(0, len(imgs), subdivision + 1, dtype=np.int)
                result_batch = torch.tensor([]).to(self.device)
                loss_batch = 0

                inner_pbar = tqdm(range(subdivision), total=subdivision, ncols=100,
                                  bar_format='{l_bar}{bar:30}{r_bar}', leave=False)

                for div in inner_pbar:
                    n, m = split[div], split[div + 1]

                    div_imgs = imgs[n:m]
                    div_labels = labels[n:m]

                    div_imgs = div_imgs.to(self.device)  # data (to gpu / cpu)
                    div_labels = div_labels.to(self.device)  # label (to gpu / cpu)

                    pred = self.net(div_imgs).to(self.device)
                    result_batch = torch.cat((result_batch, pred), dim=0)  # data into model

                    loss = self.criterion(pred, div_labels)  # calculate loss
                    loss.backward()  # calculate gradient

                    loss_batch += float(loss.item())

                self.optimizer.step()  # update parameters

                loss_batch = loss_batch / subdivision

                # tensorboard log
                self.writer.add_scalar('data/loss', loss_batch, plot_point)
                plot_point += 1

                # label
                predicted = torch.max(result_batch.data, 1)[1].cpu()  # predict
                labels = labels.cpu()
                self.all_pred = torch.cat((self.all_pred, predicted), dim=0)
                self.all_label = torch.cat((self.all_label, labels), dim=0)

                predicted = predicted.numpy()  # predict
                label_ans = labels.numpy()  # correct answer

                pred_bool = (label_ans == predicted)
                # cls_bool = [label_ans[i] for i, x in enumerate(pred_bool) if not x]
                false_step = [idx for idx, x in enumerate(pred_bool) if not x]

                for idx in false_step:
                    fp = self.false_paths[ep]
                    name = Path(paths[idx]).name

                    img_path = Path(fp, f'batch_{batch_idx}', name)
                    img_path.parent.mkdir(parents=True, exist_ok=True)

                    # save
                    # save_image(datas[idx], str(img_path))

                # release memory
                del result_batch, labels, imgs, names, paths

                # count of matched label
                cnt = pred_bool.sum()

                batch_size = len(label_ans)

                # calc total
                total_acc += cnt
                total_loss += loss_batch * batch_size

                acc = cnt / batch_size

                outer_pbar.set_description(f'TRAIN')
                outer_pbar.set_postfix(
                    ordered_dict=OrderedDict(loss=f'{loss_batch:<.6f}', acc=f'{acc:<.6f}'))

                ss = f'loss: {loss_batch:<.6f} / acc: {acc:<.6f}'.rstrip()
                # ss += f'\nans   : {label_ans_str}'
                # ss += f'\nresult: {predicted_str}'
                ss += f'\n  -> ans   : {label_ans}'
                ss += f'\n  -> result: {predicted}'

                self.log.writeline(ss, debug_ok=False)

                # break
                # end of this batch

                # imgs = imgs.to(self.device)  # data (to gpu / cpu)
                # labels = labels.to(self.device)  # label (to gpu / cpu)

                # self.optimizer.zero_grad()  # init gradient
                # out = self.net(imgs).to(self.device)  # data into model
                # loss = self.criterion(out, labels)  # calculate loss
                # loss.backward()  # calculate gradient
                # self.optimizer.step()  # update parameters

                # # tensorboard log
                # loss_val = float(loss.item())
                # self.writer.add_scalar('data/loss', loss_val, plot_point)
                # plot_point += 1

                # # label
                # predicted = torch.max(out.data, 1)[1].cpu()  # predict
                # labels = labels.cpu()
                # self.all_pred = torch.cat((self.all_pred, predicted), dim=0)
                # self.all_label = torch.cat((self.all_label, labels), dim=0)

                # predicted = predicted.numpy()  # predict
                # label_ans = labels.numpy()  # correct answer

                # pred_bool = (label_ans == predicted)
                # # cls_bool = [label_ans[i] for i, x in enumerate(pred_bool) if not x]
                # false_step = [idx for idx, x in enumerate(pred_bool) if not x]

                # for idx in false_step:
                #     fp = self.false_paths[ep]
                #     name = Path(paths[idx]).name

                #     img_path = Path(fp, f'batch_{batch_idx}', name)
                #     img_path.parent.mkdir(parents=True, exist_ok=True)

                #     # save
                #     # save_image(datas[idx], str(img_path))

                # # release memory
                # del out, labels, imgs, names, paths

                # # count of matched label
                # cnt = pred_bool.sum()

                # batch_size = len(label_ans)

                # # calc total
                # total_acc += cnt
                # total_loss += loss_val * batch_size

                # acc = cnt / batch_size

                # # look adjustment
                # # label_ans_str = ''
                # # predicted_str = ''

                # # if batch_size > 10:
                # #     # out => [x x x x x x x x ...]
                # #     label_ans_str = np.array2string(label_ans[:8])[:-1] + ' ...]'
                # #     predicted_str = np.array2string(predicted[:8])[:-1] + ' ...]'
                # # else:
                # #     # out => [x x x x x x x x x x]
                # #     label_ans_str = np.array2string(label_ans)
                # #     predicted_str = np.array2string(predicted)

                # # ss = f'ans: {label_ans_str} / result: {predicted_str}'
                # # ss += f' / loss: {loss:<8} / acc: {acc:<8}'.rstrip()

                # inner_pbar.set_description(f'TRAIN')
                # inner_pbar.set_postfix(
                #     ordered_dict=OrderedDict(loss=f'{loss_val:<.6f}', acc=f'{acc:<.6f}'))

                # ss = f'loss: {loss_val:<.6f} / acc: {acc:<.6f}'.rstrip()
                # # ss += f'\nans   : {label_ans_str}'
                # # ss += f'\nresult: {predicted_str}'
                # ss += f'\n  -> ans   : {label_ans}'
                # ss += f'\n  -> result: {predicted}'

                # self.log.writeline(ss, debug_ok=False)

                # # break
                # # end of this batch

            # add confusion matrix to tensorboard
            add_confusion_matrix_to_tensorboard(
                self.writer, self.all_label, self.all_pred, list(self.classes.values()), ep)

            # calclate total loss / accuracy
            size = len(self.train_data.dataset)
            total_loss = total_loss / size
            total_acc = total_acc / size

            # log
            self.log.writeline('\n---------------', debug_ok=False)
            self.log.writeline(f'Total loss: {total_loss}', debug_ok=False)
            self.log.writeline(f'Total acc: {total_acc}', debug_ok=False)
            self.log.writeline('---------------\n', debug_ok=False)

            # for tqdm
            print(f'  Total loss: {total_loss}')
            print(f'  Total acc: {total_acc}\n')

            self.writer.add_scalar('data/total_acc', total_acc, ep)
            self.writer.add_scalar('data/total_loss', total_loss, ep)

            # exec test cycle
            # if self.__is_execute_cycle(self.test_cycle, ep):
            if self.test_schedule[ep]:
                self.test_model.test(epoch=ep)

            # save pth cycle
            # if self.__is_execute_cycle(self.pth_save_cycle, ep):
            if self.pth_save_schedule[ep]:
                save_path = ul.create_file_path(
                    self.tms.pth_save_path, '', head=f'epoch{ep + 1}', ext='pth')

                progress = ul.ProgressLog(f'Saving model to \'{save_path}\'')
                self.save_model(save_path, _save=False)  # save
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

    def __create_schedule(self, cycle: int) -> List[bool]:
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


@dataclass(init=False)
class ValidModel(Model):
    def __init__(
            self,
            load_pth_path: str,
            use_gpu: bool = True,
            toml_settings: Optional[_tms.TomlSettings] = None,
            transform: transforms = None) -> None:

        """
        Args:
            load_pth_path (str): pth path to load.
            use_gpu (bool, optional): using gpu or cpu. Defaults to False.
            transform (transforms, optional): tensor transform. Defaults to None.
        """

        # super constructor
        # super().__init__(valid=True use_gpu=use_gpu, load_pth_path=load_pth_path)
        if toml_settings is None:
            toml_settings = _tms.factory()

        super().__init__(toml_settings=toml_settings, use_gpu=use_gpu)

        if transform is None:
            # transform
            transform = transforms.Compose([
                transforms.Resize(self.tms.input_size),
                transforms.ToTensor()])

        self.transform = transform
    # end of [function] __init__

    def valid(self, image_path: str) -> 'PredictedResult':
        """ Validing model by single image.

        Args:
            image_path (str): image path to valid.

        Raises:
            FileNotFoundError: is occured if image path is not exist.

        Returns:
            PredictedData: result of predicted.
        """

        self.net.eval()  # switch to eval

        # path is not exist -> PredictedData default value
        if not Path(image_path).exists():
            # raise FileNotFoundError
            return PredictedResult()

        # get image as tensor
        img = self.__get_image_as_tensor(image_path)

        # input to model
        x = self.net(img)
        pred = torch.max(x.data, 1)[1].cpu().numpy()
        label = pred[0]

        name = self.classes[label]

        return PredictedResult(label, name)
    # end of [function] valid

    def __get_image_as_tensor(self, image_path: str) -> torch.Tensor:
        """ Get image data as tensor type.

        Args:
            image_path (str): image path

        Returns:
            torch.Tensor: tensor of image data
        """

        img = Image.open(image_path)
        img = img.convert('RGB')

        # transform
        # if self.transform is not None:
        img = self.transform(img)

        # add fake dimension
        #   [unsqueeze] referene -> <https://pytorch.org/docs/stable/torch.html#torch.unsqueeze>
        img = torch.unsqueeze(img, 0)  # OR `img.unsqueeze_(0)`
        img = img.to(self.device)

        return img
    # end of [function] __process_image
# end of [class] ValidModel


@dataclass
class PredictedResult:
    """ Predicted result.

        Args:
            label (int, optional): label of classes. Defaults to -1.
            name (str, optional): class name. Defaults to 'None'.
            rate (float, optional): accuracy rate. Defaults to 0.0.
    """

    label: int = -1
    name: str = 'None'
    rate: float = 0.0
# end of [class] PredictedData


def add_confusion_matrix_to_tensorboard(
        writer: tbx.SummaryWriter,
        correct_labels: Union[torch.Tensor, np.ndarray],
        predicted_labels: Union[torch.Tensor, np.ndarray],
        classes: List[str],
        current_epoch: int):

    cm = calc_confusion_matrix(correct_labels, predicted_labels, len(classes))
    fig = plot_confusion_matrix(cm, classes)
    add_to_tensorboard(writer, fig, current_epoch)
# end of [function] add_confusion_matrix_to_tensorboard


def add_to_tensorboard(
        writer: tbx.SummaryWriter,
        fig: plt.figure,
        step: int):

    fig.canvas.draw()
    img = fig.canvas.renderer._renderer
    img_ar = np.array(img).transpose(2, 0, 1)

    writer.add_image('confusion matrix', img_ar)
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

    # plt.show()
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
