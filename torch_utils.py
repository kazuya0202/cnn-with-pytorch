from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

# my packages
import cnn
import utils as ul


class Data:
    def __init__(self, path: str, label: int):
        self.path = path
        self.label = label

    def items(self):
        return self.path, self.label


class CreateDataset:
    def __init__(
            self,
            path: str,
            extensions: list,
            test_size: Union[float, int]):

        self.path = path
        self.extensions = extensions
        self.test_size = test_size

        self.all_list = {}  # {'train': [], 'test': []}
        self.classes = {}  # {'label': 'class name' ...}

        # all image of each class
        # - each_cls_list[class1] = {'train': [], 'test': []}
        # - each_cls_list[class2] = {'train': [], 'test': []}
        # - ...
        self.each_cls_list = {}

        # size of images
        self.all_size = 0
        self.train_all_size = 0
        self.test_all_size = 0

        self.each_test_size = 0

        # ----------

        # train_list / test_list
        self.__get_all_datas()

        # for c in self.classes:
        #     # usage: self.classes[c] => str(label)
        #     print(self.classes[c])

    def __get_all_datas(self):
        """ Get All Datasets from each directory. """
        # init
        self.all_list = {
            'train': [],
            'test': []
        }
        path = Path(self.path)

        # directories in [image_path]
        dirs = [d for d in path.glob('*') if d.is_dir()]

        # all extensions / all sub directories
        for idx, _dir in enumerate(dirs):
            xs = []
            for ext in self.extensions:
                tmp = [Data(x.as_posix(), idx)
                       for x in _dir.glob(f'*.{ext}') if x.is_file()]
                xs.extend(tmp)

            # split dataset
            train, test = train_test_split(xs, test_size=self.test_size)
            self.all_list['train'].extend(train)
            self.all_list['test'].extend(test)

            self.each_cls_list[_dir.name] = {
                'train': train,
                'test': test
            }

            # self.classes[_dir.name] = idx
            self.classes[str(idx)] = _dir.name

        self.train_all_size = len(self.all_list['train'])
        self.test_all_size = len(self.all_list['test'])

        self.all_size = self.train_all_size + self.test_all_size

        ul.write_classes(self.classes)


class CustomDataset(Dataset):
    def __init__(self, dataset: CreateDataset, target='train', transform=None):
        self.transform = transform
        self.dataset = dataset
        self.target_list = self.dataset.all_list[target]
        self.list_size = len(self.target_list)

    def __getitem__(self, idx):
        x = self.target_list[idx]
        img = Image.open(x.path)
        img = img.convert('RGB')

        label = x.label

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.list_size


class Model:
    def __init__(
            self,
            device,
            image_size: tuple = (60, 60),
            load_pth_path: Optional[str] = None):

        self.device = device
        self.image_size = image_size

        x = self.__build_model() if load_pth_path is None \
            else self.__load_model(load_pth_path)

        self.net: cnn.Net = x[0]
        self.optimizer: Union[optim.Adam, optim.SGD] = x[1]
        self.criterion: nn.CrossEntropyLoss = x[2]

        # if load_pth_path is None:
        #     x = self.__build_model()
        #     self.net = x[0]
        #     self.optimizer: Union[optim.Adam, optim.SGD] = x[1]
        #     self.criterion = x[2]
        # else:
        #     x = self.__load_model(load_pth_path)
        #     self.net = x[0]
        #     self.optimizer: Union[optim.Adam, optim.SGD] = x[1]
        #     self.criterion = x[3]

    def __build_model(self):
        net = cnn.Net(input_size=self.image_size)  # network

        # optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        net.to(self.device)  # switch to GPU / CPU
        net.zero_grad()  # init all gradient

        return net, optimizer, criterion

    def __load_model(self, path: str):
        checkpoint = torch.load(path)

        net = cnn.Net(input_size=self.image_size)
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion = nn.CrossEntropyLoss()

        net.to(self.device)

        return net, optimizer, criterion
        # return net, optimizer, criterion, epoch

    def save_model(self, path: str, epoch: Optional[int] = None):
        torch.save({
            # 'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'criterion': type(self.criterion).__name__
        }, path)


class TestModel:
    def __init__(
            self,
            model: Model,
            test_data: DataLoader,
            classes: Optional[dict] = None,
            log_file: ul.LogFile = ul.LogFile(None)):

        self.model = model
        self.test_data = test_data

        # classes
        self.classes = classes if classes is not None \
            else ul.load_classes()

        self.log_f = log_file

    def test(self):
        # switch to test
        self.model.net.eval()

        self.log_f.writeline('Start test.\n')
        device = self.model.device  # device
        total_acc = 0  # total accuracy

        _dict = dict([(_cls, 0) for _cls in self.classes.keys()])
        acc_list = _dict.copy()  # acc of each class
        all_size = _dict.copy()  # test size of each class

        print('running ', end='')

        # test
        for data, label in self.test_data:
            print('.', end='')

            data = data.to(device)
            label = label.to(device)

            # data into model
            out = self.model.net(data).to(device)

            # label
            predicted = torch.max(out.data, 1)[1].cpu().numpy()  # predict
            label_ans = label.cpu().numpy()  # correct answer

            # label of correct answer
            # count of matched label, and it add to total_acc
            for k in acc_list.keys():
                t = int(k)

                # match -> 1, else -> -1 / -2
                x = np.where(label_ans == t, 1, -1)
                y = np.where(predicted == t, 1, -2)

                # print(k, (x == y).sum(), x.tolist().count(1))
                acc_list[k] += (x == y).sum()  # match -> correct
                all_size[k] += x.tolist().count(1)  # all size of each class
        print()

        # calc each accuracy
        for k, _cls in self.classes.items():
            acc = round(acc_list[k] / all_size[k], 4)
            total_acc += acc

            _cls = f'[{_cls}]'
            self.log_f.writeline(
                f'{_cls:<12} acc: {acc:<6} ({acc_list[k]} / {all_size[k]} images.)')

        # total accuracy
        total_acc /= len(self.classes)
        self.log_f.writeline(f'\nTotal acc: {total_acc}')


class TrainModel:
    def __init__(
            self,
            model: Model,
            epoch: int,
            train_data: DataLoader,
            test_data: Optional[DataLoader] = None,
            classes: Optional[dict] = None,
            log_file: ul.LogFile = ul.LogFile(None),
            rate_file: ul.LogFile = ul.LogFile(None),
            is_test_per_epoch: bool = True,
            pth_save_cycle: int = 0,
            pth_epoch_path: Optional[str] = None):

        # parameters
        self.model = model
        self.epoch = epoch
        self.train_data = train_data
        self.test_data = test_data
        self.pth_save_cycle = pth_save_cycle

        # save model path
        self.pth_epoch_path = './' if pth_epoch_path is None \
            else pth_epoch_path

        # classes
        self.classes = classes if classes is not None \
            else ul.load_classes()

        self.log_f = log_file
        self.rate_f = rate_file

        self.is_test_per_epoch = is_test_per_epoch

        if self.test_data is None:
            self.is_test_per_epoch = False
        else:
            self.test_model = TestModel(
                self.model, self.test_data, self.classes, self.log_f)

    # from pytorch_memlab import profile
    # @profile
    def train(self):
        self.log_f.writeline('Start training.')
        device = self.model.device

        # switch to train
        self.model.net.train()

        # loop epoch
        for ep in range(self.epoch):
            total_loss = 0  # total loss
            total_acc = 0  # total accuracy

            torch.cuda.empty_cache()
            self.log_f.writeline(f'\n----- Epoch: {ep + 1} -----')

            # batch process
            for data, label in self.train_data:
                data = data.to(device)  # data (to gpu / cpu)
                label = label.to(device)  # label (to gpu / cpu)

                self.model.optimizer.zero_grad()  # init gradient
                out = self.model.net(data).to(device)  # data into model
                loss = self.model.criterion(out, label)  # calculate loss
                loss.backward()  # calculate gradient
                self.model.optimizer.step()  # update parameters

                # label
                predicted = torch.max(out.data, 1)[1].cpu().numpy()  # predict
                label_ans = label.cpu().numpy()  # correct answer

                # count of matched label
                cnt = (label_ans == predicted).sum()

                # calc total
                batch_size = len(label_ans)
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
                    label_ans_str = np.array2string(label_ans[:8])[:-1] + ' ...]'
                    predicted_str = np.array2string(predicted[:8])[:-1] + ' ...]'
                else:
                    # out => [x x x x x x x x x x]
                    label_ans_str = np.array2string(label_ans)
                    predicted_str = np.array2string(predicted)

                # log
                ss = f'ans: {label_ans_str} / result: {predicted_str}'
                ss += f' / loss: {loss:<8} / acc: {acc:<8}'.rstrip()
                self.log_f.writeline(ss)

                # break
                # end this batch

            # calclate total loss / accuracy
            total_loss = total_loss / len(self.train_data.dataset)
            total_acc = total_acc / len(self.train_data.dataset)

            # log
            self.log_f.writeline('\n----------')
            self.log_f.writeline(f'Total loss: {total_loss}')
            self.log_f.writeline(f'Total acc: {total_acc}')
            self.log_f.writeline('----------\n')

            torch.cuda.empty_cache()

            if self.is_test_per_epoch:
                self.test_model.test()  # test

            # save pth
            if self.pth_save_cycle != 0:
                if (ep + 1) % self.pth_save_cycle == 0:
                    self.model.save_model(self.pth_epoch_path)

            torch.cuda.empty_cache()

            # break
            # end of this epoch
        # end all epoch
