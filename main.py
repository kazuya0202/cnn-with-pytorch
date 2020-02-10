# from datetime import datetime
from pathlib import Path

import numpy as np
# import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import global_variables as _gv
# my packages
import torch_utils as tu
import utils as ul


class Main:
    def __init__(self):
        pass
    # end of [function] __init__

    def execute(self):
        gv = _gv.GlobalVariables()

        """ DEBUG NOW """
        # gv.is_save_debug_log = False
        # gv.is_save_rate_log = False
        """ --------- """

        # if not exist, exit script
        if not Path(gv.image_path).exists():
            print(f'The directory \'{gv.image_path}\' does not exist.')
            exit(-1)

        # ===== log / rate file =====
        logs = ul.DebugRateLogs()  # init as None

        # update debug log
        if gv.is_save_debug_log:
            p = ul.create_file_path(gv.log_path, gv.filename_base)
            logs.set_log(ul.LogFile(p, default_debug_ok=True))

        # update rate log
        if gv.is_save_rate_log:
            p = ul.create_file_path(gv.log_path, gv.filename_base, ext='csv')
            logs.set_rate(ul.LogFile(p, default_debug_ok=True))

        # ===== create datasets =====
        progress = ul.ProgressLog(f'Create dataset from \'{gv.image_path}\'')

        # get input image size as tuple
        image_size = gv.image_size
        if not isinstance(image_size, tuple):
            image_size = (image_size, image_size)

        # transform
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()])

        # train, test(unknown), test(known)
        dataset = tu.CreateDataset(
            path=gv.image_path,
            extensions=gv.extensions,
            test_size=gv.test_size)

        train_data, unknown_data, known_data = self.__create_custom_dataloader(
            dataset=dataset,
            batch_size=gv.batch_size,
            transform=transform,
            is_shuffle=gv.is_shuffle_per_epoch)

        # train_data, unknown_data = self.__create_mnist_dataloader(
        #     # dataset=dataset,
        #     batch_size=gv.batch_size,
        #     transform=transform,
        #     is_shuffle=gv.is_shuffle_per_epoch)
        # known_data = None

        progress.complete()

        # ===== create make required direcotry =====
        progress = ul.ProgressLog('Making required directory')

        # arguments of required path
        params = [gv.false_path,
                  gv.log_path,
                  gv.pth_path]

        ul.make_directories(*params)
        progress.complete()

        # ===== network =====
        progress = ul.ProgressLog('Building CNN network')  # debug log
        # create network
        # model = tu.Model(
        #     use_gpu=gv.use_gpu,
        #     image_size=image_size,
        #     logs=logs)
        model = tu.Model(
            classes=dataset.classes,
            use_gpu=gv.use_gpu,
            image_size=image_size,
            logs=logs)

        progress.complete()

        print('Classify classes:')
        for label, _cls in model.classes.items():
            print(f'  {label}: {_cls}')
        print()

        # ===== dataset model =====
        test_model = tu.TestModel(model, unknown_data, known_data)

        train_model = tu.TrainModel(
            model=model,
            epoch=gv.epoch,
            train_data=train_data,
            test_model=test_model,
            gv=gv)

        # exec
        train_model.train()  # train
        test_model.test()  # final test

        # ===== save model =====
        # pth path
        pt_params = [gv.pth_path, gv.filename_base]
        p = ul.create_file_path(*pt_params, end='_final', ext='pth')

        progress = ul.ProgressLog(f'Saving model to \'{p}\'')
        # train_model.model.save_model(p)  # save
        logs.log.writeline(f'Saved model to \'{p}\'', debug_ok=False)
        progress.complete()
    # end of [function] execute

    def __create_custom_dataloader(
            self,
            dataset: tu.CreateDataset,
            batch_size: int,
            transform: transforms,
            is_shuffle: bool):

        # train dataset / test dataset
        train_dataset = tu.CustomDataset(dataset, 'train', transform)
        unknown_dataset = tu.CustomDataset(dataset, 'unknown', transform)
        known_dataset = tu.CustomDataset(dataset, 'known', transform)

        # batch size for testing
        test_batch = int(np.ceil(batch_size / 10))
        if test_batch < 1:
            test_batch = 1

        # train dataloader / test dataloader, shuffle
        train_data = DataLoader(train_dataset, batch_size, is_shuffle)
        unknown_data = DataLoader(unknown_dataset, test_batch, is_shuffle)
        known_data = DataLoader(known_dataset, test_batch, is_shuffle)

        return (train_data, unknown_data, known_data)
    # end of [function] __create_custom_dataset

    def __create_mnist_dataloader(
            self,
            batch_size: int,
            transform: transforms,
            is_shuffle: bool):

        train_dataset = torchvision.datasets.MNIST(
            root='./mnist-data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(
            root='/mnist-data', train=False, download=True, transform=transform)

        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=is_shuffle)
        test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=is_shuffle)

        return train_data, test_data

    def print_parameter_config(self):
        return
        # cls_num = len(self.classes)

        # _cls = list(self.classes.keys())[0]
        # each_cls_size = self.each_cls_list[_cls]['train']
        # each_cls_size += self.each_cls_list[_cls]['test']

        print('----- Configs -----')
        print(f'* image path: {self.path}')
        print(f'* extensions: {self.extensions}')
        # print(f'* dataset size: {self.all_size} ({each_cls_size} * {cls_num} = {self.all_size})')
        print(f'* ')
        print(f'* ')

        # print(f'* image path:', self.path)
        # print(f'* extension:', self.extensions)
        # print(f'* image size:', self.all_size)
        # print(f'* train size:', self.train_size)
        # print(f'* test size:', self.test_size)

        print(f'\n{self.path}')
        for i, x in enumerate(self.classes):
            print(f'    |- {x} : label [{i}]')
        print('--------------------------------\n')
    # end of [function] print_parameter_config
# end of [class] Main


if __name__ == "__main__":
    main = Main()
    main.execute()
