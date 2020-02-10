import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorboardX as tbx
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# my packages
import cnn
import global_variables as _gv
import utils as ul


@dataclass
class Data:
    path: str  # path
    label: int  # label
    name: str  # name

    # def __init__(self, path: str, label: int, name: str):
    #     self.path = path  # path
    #     self.label = label  # label
    #     self.name = name  # class name
    # # end of [function] __init__

    def items(self):
        return self.path, self.label, self.name
    # end of [function] items
# end of [class] Data


@dataclass(init=False)
class CreateDataset:
    def __init__(
            self,
            path: str,
            extensions: list,
            test_size: Union[float, int]):

        self.path: str = path
        self.extensions: List[str] = extensions
        self.test_size: Union[float, int] = test_size

        self.all_list: Dict[str, list] = {}  # {'train': [], 'unknown': [], 'known': []}
        self.classes: Dict[int, str] = {}  # {label: 'class name' ...}

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

    def __write_config(self):
        _dir = Path('config')
        _dir.mkdir(parents=True, exist_ok=True)

        # # -- write classes
        # path = _dir.joinpath('classes.txt')
        # with open(path, 'w') as f:
        #     for k, _cls in self.classes.items():
        #         f.write(f'{k}:{_cls}\n')

        # # -- train image path
        path = _dir.joinpath('train_used_images.txt')
        with open(path, 'w') as f:
            f.write('--- Image used for training. ---\n')

            for x in self.all_list['train']:
                p = Path(x.path).absolute()
                f.write(p.as_posix() + '\n')

        # # -- unknown image path
        path = _dir.joinpath('unknown_used_images.txt')
        with open(path, 'w') as f:
            f.write('--- Image used for unknown testing. ---\n')

            for x in self.all_list['unknown']:
                p = Path(x.path).absolute()
                f.write(p.as_posix() + '\n')

        # # -- known image path
        path = _dir.joinpath('known_used_images.txt')
        with open(path, 'w') as f:
            f.write('--- Image used for known testing. ---\n')

            for x in self.all_list['known']:
                p = Path(x.path).resolve()
                f.write(p.as_posix() + '\n')
    # end of [function] __write_config

    def __get_all_datas(self):
        """ Get All Datasets from each directory. """
        self.all_list = {
            'train': [],
            'unknown': [],
            'known': [],
        }  # init
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

            # self.classes[_dir.name] = idx
            self.classes[idx] = _dir.name

        self.train_size = len(self.all_list['train'])
        self.unknown_size = len(self.all_list['unknown'])
        self.known_size = len(self.all_list['known'])

        self.all_size = self.train_size + self.unknown_size
    # end of [function] __get_all_datas
# end of [class] CreateDataset


@dataclass(init=False)
class CustomDataset(Dataset):
    def __init__(self, dataset: CreateDataset, target='train', transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_list = self.dataset.all_list[target]
        self.list_size = len(self.target_list)
    # end of [function] __init__

    def __getitem__(self, idx):
        x = self.target_list[idx]
        path, label, name = x.items()

        img = Image.open(path)
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label, name
    # end of [function] __getitem__

    def __len__(self):
        return self.list_size
    # end of [function] __len__
# end of [class] CustomDataset


@dataclass(init=False)
class Model:
    def __init__(
            self,
            # dataset: CreateDataset,
            classes: Optional[dict] = None,
            use_gpu: bool = False,
            image_size: Tuple[int, int] = (60, 60),
            logs: Optional[ul.DebugRateLogs] = None,
            load_pth_path: Optional[str] = None) -> None:
        """
        `load_pth_path` is not None -> load pth automatically.
        """

        # network
        self.net: cnn.Net
        self.optimizer: Union[optim.Adam, optim.SGD]
        self.criterion: nn.CrossEntropyLoss

        self.classes: dict  # class
        self.image_size = image_size

        # gpu setting
        self.use_gpu = torch.cuda.is_available() and use_gpu
        self.device = torch.device('cuda' if use_gpu else 'cpu')

        # tensorboard
        self.writer = tbx.SummaryWriter()

        # assign or load from classes.txt
        if load_pth_path is None:
            self.classes = classes if classes is not None \
                else ul.load_classes().copy()

        # assign or create instance
        self.logs = logs if logs is not None \
            else ul.DebugRateLogs()

        # build model by cnn.py or load model
        self.__build_model(load_pth_path)

        # save classes
        self.__write_classes()

        # classes.txt is EOF, classes is None
        if self.classes is {}:
            print('classes is {}')
    # end of __init__()

    def __build_model(self, load_pth_path: Optional[str]):
        self.net = cnn.Net(input_size=self.image_size)  # network

        # optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=pow(10, -8))
        self.criterion = nn.CrossEntropyLoss()

        # is load
        if load_pth_path is None:
            self.net.zero_grad()  # init all gradient
        else:
            # check exist
            if not Path(load_pth_path).exists():
                print(f'{load_pth_path} is not exist.')

            # load checkpoint
            checkpoint = torch.load(load_pth_path)

            # classes, network
            self.classes = checkpoint['classes']
            self.net.load_state_dict(checkpoint['model_state_dict'])

            # epoch = checkpoint['epoch']
            # criterion = checkpoint['criterion']
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.net.to(self.device)  # switch to GPU / CPU
    # end of __build_model()

    def save_model(self, path: str, epoch: Optional[int] = None) -> None:
        torch.save({
            'classes': self.classes,
            'model_state_dict': self.net.state_dict(),

            # 'epoch': epoch,
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            # 'criterion': type(self.criterion).__name__
        }, path)
    # end of [function] save_model

    def __write_classes(self):
        _dir = Path('config')
        _dir.mkdir(parents=True, exist_ok=True)

        # -- write classes
        path = _dir.joinpath('classes.txt')
        with open(path, 'w') as f:
            for k, _cls in self.classes.items():
                f.write(f'{k}:{_cls}\n')
    # end of [function] __write_classes
# end of [class] Model


@dataclass(init=False)
class TestModel:
    def __init__(
            self,
            model: Model,
            unknown_data: DataLoader,
            known_data: DataLoader = None):

        self.model = model
        self.unknown_data = unknown_data
        self.known_data = known_data

        self.log = model.logs.log  # log
        self.rate = model.logs.rate  # rate
    # end of [function] __init__

    def test(self, epoch=None):
        # switch to test
        self.model.net.eval()

        self.log.writeline('# Start testing.\n', False)
        device = self.model.device  # device

        def __execute(data_loader: DataLoader, tqdm_desc: str = ''):
            total_acc = 0  # total accuracy

            _dict = dict([(_cls, 0) for _cls in self.model.classes.keys()])
            acc_list = _dict.copy()  # acc of each class
            all_size = _dict.copy()  # test size of each class

            # test
            # pbar = tqdm(self.unknown_data, total=len(self.unknown_data),
            pbar = tqdm(data_loader, total=len(data_loader),
                        ncols=100, bar_format='{l_bar}{bar:30}{r_bar}')
            pbar.set_description(tqdm_desc)

            # for data, label, name in self.test_data:
            for datas, labels, name in pbar:
                datas = datas.to(device)
                labels = labels.to(device)

                # data into model
                out = self.model.net(datas).to(device)

                # label
                predicted = torch.max(out.data, 1)[1].cpu().numpy()  # predict
                label_ans = labels.cpu().numpy()  # correct answer

                # label of correct answer
                # count of matched label, and it add to total_acc
                for k in acc_list.keys():
                    t = int(k)

                    # match -> 1, else -> -1 / -2
                    x = np.where(label_ans == t, 1, -1)
                    y = np.where(predicted == t, 1, -2)

                    acc_list[k] += (x == y).sum()  # match -> correct
                    all_size[k] += x.tolist().count(1)  # all size of each class
            # end of all test

            # calc each accuracy
            for k, _cls in self.model.classes.items():
                acc = round(acc_list[k] / all_size[k], 4)
                total_acc += acc

                _cls = f'[{_cls}]'

                print('  ', end='')
                ss = f'{_cls:<12} -> acc: {acc:<6} ({acc_list[k]} / {all_size[k]} images.)'
                self.log.writeline(ss)
            # end of calculate accuracy of each class and total
            self.log.writeline()

            # total accuracy
            total_acc /= len(self.model.classes)
            print('  ', end='')
            self.log.writeline(f'Total acc: {total_acc}\n')

            self.model.writer.add_scalar('test/total_acc', total_acc)
        # end of [function] __execute

        __execute(self.unknown_data, 'TEST [unknonw]')  # unknown test

        if self.known_data is not None:
            __execute(self.known_data, 'TEST [ known ]')  # knwon test
    # end of [function] test
# end of [class] TestMedel


@dataclass(init=False)
class TrainModel():
    def __init__(
            self,
            model: Model,
            epoch: int,
            train_data: DataLoader,
            test_model: TestModel,
            gv: Optional[_gv.GlobalVariables] = None):

        # parameters
        self.model = model  # Model
        self.epoch = epoch
        self.train_data = train_data
        self.test_model = test_model  # TestModel

        self.log = model.logs.log  # log
        self.rate = model.logs.rate  # rate

        if gv is None:
            gv = _gv.GlobalVariables()

        # cycle
        self.test_cycle = gv.test_cycle
        self.pth_save_cycle = gv.pth_save_cycle

        # is save
        if gv.pth_save_cycle != 0:
            # path of model following save cycle
            self.pth_epoch_path = Path(gv.pth_path, 'epoch_pth', gv.filename_base)
            self.pth_epoch_path.mkdir(parents=True, exist_ok=True)

    # end [function] __init__

    def train(self):
        self.log.writeline('# Start training.', False)
        is_availabled = torch.cuda.is_available()
        self.log.writeline(f'# GPU -> available: {is_availabled}, used: {self.model.use_gpu}')

        device = self.model.device

        # switch to train
        self.model.net.train()

        plot_point = 0  # for tensorboard

        # loop epoch
        for ep in range(self.epoch):
            total_loss = 0  # total loss
            total_acc = 0  # total accuracy

            self.log.writeline(f'----- Epoch: {ep + 1} -----')

            # batch process
            pbar = tqdm(enumerate(self.train_data), total=len(self.train_data),
                        ncols=100, bar_format='{l_bar}{bar:30}{r_bar}')

            for batch_idx, (datas, labels, names) in pbar:
                datas = datas.to(device)  # data (to gpu / cpu)
                labels = labels.to(device)  # label (to gpu / cpu)

                self.model.optimizer.zero_grad()  # init gradient
                out = self.model.net(datas).to(device)  # data into model
                loss = self.model.criterion(out, labels)  # calculate loss
                loss.backward()  # calculate gradient
                self.model.optimizer.step()  # update parameters

                batch_size = len(labels)

                # tensorboard log
                self.model.writer.add_scalar('data/loss', loss.item(), plot_point)
                plot_point += 1

                # label
                predicted = torch.max(out.data, 1)[1].cpu().numpy()  # predict
                label_ans = labels.cpu().numpy()  # correct answer

                # count of matched label
                cnt = (label_ans == predicted).sum()

                # calc total
                total_acc += cnt
                total_loss += loss * batch_size

                # round
                acc = round(cnt / batch_size, 6)
                loss = round(float(loss.item()), 6)

                label_ans_str = ''
                predicted_str = ''

                # look adjustment
                if batch_size > 10:
                    # out => [x x x x x x x x ...]
                    # label_ans_str = np.array2string(label_ans[:8])[:-1] + ' ...]'
                    # predicted_str = np.array2string(predicted[:8])[:-1] + ' ...]'
                    label_ans_str = np.array2string(label_ans)
                    predicted_str = np.array2string(predicted)

                else:
                    # out => [x x x x x x x x x x]
                    label_ans_str = np.array2string(label_ans)
                    predicted_str = np.array2string(predicted)

                pbar.set_description(f'TRAIN')
                pbar.set_postfix(
                    ordered_dict=OrderedDict(loss=f'{loss:<8}', acc=f'{acc:<8}'))

                # log
                # ss = f'ans: {label_ans_str} / result: {predicted_str}'
                # ss += f' / loss: {loss:<8} / acc: {acc:<8}'.rstrip()

                ss = f'loss: {loss:<8} / acc: {acc:<8}'.rstrip()
                ss += f' / ans: {label_ans_str} / result: {predicted_str}'
                self.log.writeline(ss, debug_ok=False)

                # break
                # end of this batch

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
            print(f'  Total acc: {total_acc}')
            print()

            self.model.writer.add_scalar('data/total_acc', total_acc, ep)
            self.model.writer.add_scalar('data/total_loss', total_loss, ep)

            # exec test
            if self.test_cycle != 0:
                # cycle / not last epoch -> exec
                if (ep + 1) % self.test_cycle == 0 and (ep + 1) != self.epoch:
                    self.test_model.test(epoch=ep)  # testing

            # save pth
            if self.pth_save_cycle != 0:
                # cycle / not last epoch -> exec
                if (ep + 1) % self.pth_save_cycle == 0 and (ep + 1) != self.epoch:

                    pt_params = [self.pth_epoch_path, '']
                    p = ul.create_file_path(*pt_params, head=f'epoch{ep + 1}', ext='pth')

                    progress = ul.ProgressLog(f'Saving model to \'{p}\'')
                    # self.model.save_model(p)  # save
                    progress.complete()

                    # log
                    self.log.writeline(f'# Saved model to \'{p}\'', debug_ok=False)

            # break
            # end of this epoch

        # export as json
        self.model.writer.export_scalars_to_json('all_scalars.json')
        self.model.writer.close()

        # end of all epoch
    # end of [function] train
# end of [class] TrainModel


@dataclass(init=False)
class ValidModel(Model):
    # def __init__(self, model: Model):
    #     self.model = model
    def __init__(
            self,
            load_pth_path: str,
            use_gpu: bool = False,
            transform=None):

        super().__init__(use_gpu=True, load_pth_path=load_pth_path)

        if transform is None:
            # transform
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()])

        self.transform = transform
    # end of [function] __init__

    def valid(self, image_path: str):
        self.net.eval()  # switch to eval

        # not exist -> -1
        if not Path(image_path).exists():
            return -1

        # get image to tensor
        #   [unsqueeze] referene -> <https://pytorch.org/docs/stable/torch.html#torch.unsqueeze>
        img = self.__get_image_as_tensor(image_path)
        img = torch.unsqueeze(img, 0)  # OR img.unsqueeze_(0)
        img = img.to(self.device)

        x = self.net(img)
        pred = torch.max(x.data, 1)[1].cpu().numpy()

        label = pred[0]
        return label
    # end of [function] valid

    def __get_image_as_tensor(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path)
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img
    # end of [function] __process_image
# end of [class] ValidModel
