from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# my packages
import torch_utils as tu
import utils as ul
import global_variables as _gv


class Main:
    def __init__(self):
        # GPU / CPU
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    # end of [function] __init__

    def execute(self):
        gv = _gv.GlobalVariables()

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
            p = ul.create_file_path(gv.log_path, gv.filename_base)
            logs.set_rate(ul.LogFile(p, default_debug_ok=True))

        # ===== create datasets =====
        debug_log = ul.DebugLog(f'Create dataset from \'{gv.image_path}\'')

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
            transform=transform,
            is_shuffle=gv.is_shuffle_per_epoch)

        debug_log.complete()

        # classes
        for k, _cls in dataset.classes.items():
            logs.log.writeline(f'{k}: {_cls}')
        logs.log.writeline()

        # ===== create make required direcotry =====
        debug_log = ul.DebugLog('Making required directory')

        # arguments of required path
        params = [
            gv.false_path,
            gv.log_path,
            gv.pth_path]

        ul.make_directories(*params)
        debug_log.complete()

        # ===== network =====
        debug_log = ul.DebugLog('Building CNN network')  # debug log

        model = tu.Model(self.device, dataset.classes, image_size)

        debug_log.complete()

        # ===== dataset model =====
        test_model = tu.TestModel(
            model=model,
            test_data=test_data,
            logs=logs)

        train_model = tu.TrainModel(
            model=model,
            epoch=gv.epoch,
            train_data=train_data,
            test_model=test_model,
            gv=gv,
            # test_cycle=gv.test_cycle,
            # pth_save_cycle=gv.pth_save_cycle,
            # pth_epoch_path=pth_epoch_path,
            logs=logs)

        # exec
        train_model.train()  # train
        test_model.test()  # final test

        # ===== save model =====
        # pth path
        pt_params = [gv.pth_path, gv.filename_base]
        p = ul.create_file_path(*pt_params, ext='pth')

        debug_log = ul.DebugLog(f'Saving model to \'{p}\'')
        # train_model.model.save_model(p)  # save
        logs.log.writeline(f'Saved model to \'{p}\'')
        debug_log.complete()
    # end of [function] execute

    def __create_custom_dataset(
            self,
            dataset: tu.CreateDataset,
            batch_size: int,
            transform: transforms,
            is_shuffle: bool):

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
    # end of [function] __create_custom_dataset

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
