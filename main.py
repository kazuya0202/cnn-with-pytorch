import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# my packages
import torch_utils as tu
import utils as ul
from global_variables import GlobalVariables


class Main:
    def __init__(self):
        # GPU / CPU
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # get file name
        self.filename_base = str(datetime.datetime.now().strftime(
            "ymd%Y%m%d_hms%H%M%S")).replace(" ", "_")

    def execute(self):
        # import cupy as cp
        # pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        # cp.cuda.set_allocator(pool.malloc)

        gv = GlobalVariables()

        # if not exist, exit script
        if not Path(gv.image_path).exists():
            print(f'The directory \'{gv.image_path}\' does not exist.')
            exit(-1)

        # ===== log file =====
        # parameters for LogFile
        log_params = [gv.log_path, self.filename_base]

        # log file
        p = None if not gv.is_save_debug_log \
            else ul.create_file_path(*log_params)
        log_file = ul.LogFile(p, default_debug_ok=True)

        # rate file
        p = None if not gv.is_save_rate_log \
            else ul.create_file_path(*log_params, ext='csv')
        rate_file = ul.LogFile(p, default_debug_ok=True)

        # ===== create datasets =====
        log = ul.DebugLog(f'Create dataset from \'{gv.image_path}\'')

        # get input image size as tuple
        image_size = gv.image_size
        if not isinstance(image_size, tuple):
            image_size = (image_size, image_size)

        # transform
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()])

        dataset = tu.CreateDataset(
            path=gv.image_path,
            extensions=gv.extensions,
            test_size=gv.test_size)

        train_data, test_data = self.__create_custom_dataset(
            dataset=dataset,
            batch_size=gv.batch_size,
            transform=transform)

        log.complete()

        # classes
        for k, _cls in dataset.classes.items():
            log_file.writeline(f'{k}: {_cls}')
        log_file.writeline()

        # ===== create make required direcotry =====
        log = ul.DebugLog('Making required directory')

        pth_epoch_path = None if gv.pth_save_cycle == 0 \
            else f'{gv.pth_path}/epoch_pth'

        # arguments of required path
        params = [
            gv.false_path,
            gv.log_path,
            gv.pth_path,
            pth_epoch_path]

        ul.make_directories(*params)
        log.complete()

        # ===== network =====
        log = ul.DebugLog('Create CNN network')  # debug log

        model = tu.Model(self.device, image_size)

        log.complete()

        # ===== train =====
        train_model = tu.TrainModel(
            model=model,
            epoch=gv.epoch,
            train_data=train_data,
            test_data=test_data,
            classes=dataset.classes,
            log_file=log_file,
            rate_file=rate_file,
            is_test_per_epoch=gv.is_test_per_epoch,
            pth_save_cycle=gv.pth_save_cycle,
            pth_epoch_path=pth_epoch_path)

        train_model.train()

        # ===== final test =====
        test_model = tu.TestModel(
            model=model,
            test_data=test_data,
            classes=dataset.classes,
            log_file=log_file)

        test_model.test()

        # ===== save model =====
        pt_params = [gv.pth_path, self.filename_base]
        p = ul.create_file_path(*pt_params, ext='pth')

        log = ul.DebugLog(f'Saving model {p}')

        train_model.model.save_model(p)

        log.complete()

    def __create_custom_dataset(
            self,
            dataset: tu.CreateDataset,
            batch_size: int,
            transform: transforms):

        # train dataset, shuffle
        train_dataset = tu.CustomDataset(
            dataset=dataset,
            target='train',
            transform=transform)

        train_data = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True)

        # test dataset, shuffle
        test_dataset = tu.CustomDataset(
            dataset=dataset,
            target='test',
            transform=transform)

        batch = int(np.ceil(batch_size / 10.0))

        test_data = DataLoader(
            test_dataset,
            batch_size=batch,
            shuffle=True)

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


if __name__ == "__main__":
    main = Main()
    main.execute()
