import errno
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# my packages
import global_variables as _gv
import torch_utils as tu
import utils as ul


@dataclass
class Main:
    def execute(self):
        gv = _gv.GlobalVariables()

        """ DEBUG NOW """
        gv.is_save_debug_log = False
        gv.is_save_rate_log = False
        """ --------- """

        # if not exist, exit script
        if not Path(gv.image_path).exists():
            err = OSError(errno.ENOENT, os.strerror(errno.ENOENT), gv.image_path)
            raise FileNotFoundError(err)

        # ===== log / rate file =====
        log = ul.LogFile(None)
        rate = ul.LogFile(None)

        if gv.is_save_debug_log:
            p = ul.create_file_path(gv.log_path, gv.filename_base)
            log = ul.LogFile(p, default_debug_ok=True)

        if gv.is_save_rate_log:
            p = ul.create_file_path(gv.log_path, gv.filename_base, ext='csv')
            rate = ul.LogFile(p, default_debug_ok=True)

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
            batch_size=gv.mini_batch,
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

        # save pth in any epoch
        pth_save_path = None if gv.pth_save_cycle == 0 \
            else Path(gv.pth_save_path, 'epoch_pth', gv.filename_base).as_posix()

        # arguments of required path
        params = [gv.false_path,
                  gv.log_path,
                  gv.pth_save_path,
                  pth_save_path]

        ul.make_directories(*params)
        progress.complete()

        # ===== network =====
        progress = ul.ProgressLog('Building CNN network')  # debug log
        # create network
        model = tu.Model(
            classes=dataset.classes,
            use_gpu=gv.use_gpu,
            image_size=image_size,
            log=log,
            rate=rate)

        progress.complete()

        # logging parameter config
        if gv.is_print_params:
            self.__print_parameter_config(gv, model, dataset, log)

        # ===== dataset model =====
        test_model = tu.TestModel(model, unknown_data, known_data)

        train_model = tu.TrainModel(
            model=model,
            epoch=gv.epoch,
            train_data=train_data,
            test_model=test_model,
            test_cycle=gv.test_cycle,
            pth_save_cycle=gv.pth_save_cycle,
            pth_save_path=pth_save_path,
            false_path=gv.false_path)

        # exec
        train_model.train()  # train
        test_model.test(epoch=gv.epoch - 1)  # final test

        # ===== save model =====
        # pth path
        pt_params = [gv.pth_save_path, gv.filename_base]
        save_path = ul.create_file_path(*pt_params, end='_final', ext='pth')

        progress = ul.ProgressLog(f'Saving model to \'{save_path}\'')
        if gv.is_save_final_pth:
            train_model.save_model(save_path, _save=True)  # save
        progress.complete()

        log.writeline(f'# Saved model to \'{save_path}\'', debug_ok=False)
    # end of [function] execute

    def __create_custom_dataloader(
            self,
            dataset: tu.CreateDataset,
            batch_size: int,
            transform: transforms,
            is_shuffle: bool) -> tuple:

        # train dataset / test dataset
        train_dataset = tu.CustomDataset(dataset, 'train', transform)
        unknown_dataset = tu.CustomDataset(dataset, 'unknown', transform)
        known_dataset = tu.CustomDataset(dataset, 'known', transform)

        # batch size for testing
        test_batch = int(np.ceil(batch_size / 10))
        if test_batch < 1:
            test_batch = 1

        # train dataloader / test dataloader, shuffle
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=is_shuffle)
        unknown_data = DataLoader(unknown_dataset, batch_size=test_batch, shuffle=is_shuffle)
        known_data = DataLoader(known_dataset, batch_size=test_batch, shuffle=is_shuffle)

        return train_data, unknown_data, known_data
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

    def __print_parameter_config(
            self,
            gv: _gv.GlobalVariables,
            model: tu.Model,
            dataset: tu.CreateDataset,
            log: ul.LogFile) -> None:

        global_conf = {
            'runtime': gv.filename_base,
            'image path': gv.image_path,
            'supported extensions': gv.extensions,
            'is save debug log': gv.is_save_debug_log,
            'is save rate log': gv.is_save_rate_log,
            'pth save cycle': gv.pth_save_cycle,
            'test cycle': gv.test_cycle,
            'is save final pth': gv.is_save_final_pth,
            # is gradcam...
        }

        dataset_conf = {
            'limit dataset size': gv.limit_dataset_size,
            'train dataset size': len(dataset.all_list['train']),
            'unknown dataset size': len(dataset.all_list['unknown']),
            'known dataset size': len(dataset.all_list['known']),
        }

        is_available = torch.cuda.is_available()
        model_conf = {
            'net': model.net.__str__(),
            'optimizer': model.optimizer.__str__(),
            'criterion': model.criterion.__str__(),
            'input size': model.image_size,
            'epoch': gv.epoch,
            'GPU': f'available: {is_available}, used: {model.use_gpu}'
        }

        def __inner_execute(_dict: Dict[str, Any], head: str = ''):
            log.writeline(head)

            # adjust to max length of key
            max_len = max([len(x) for x in _dict.keys()])
            _format = f'%-{max_len}s : %s'

            for k, v in _dict.items():
                # format for structure of network
                if isinstance(v, str) and v.find('\n') > -1:
                    v = v.replace('\n', '\n' + ' ' * (max_len + 3)).rstrip()

                log.writeline(_format % (k, v))
            log.writeline('\n')

        log.writeline('--- Classify Classes ---')
        for label, _cls in model.classes.items():
            log.writeline(f'  {label}: {_cls}')
        log.writeline('\n')

        __inner_execute(global_conf, '--- Global Config ---')
        __inner_execute(dataset_conf, '--- Dataset Config ---')
        __inner_execute(model_conf, '--- Model Config ---')
    # end of [function] __print_parameter_config
# end of [class] Main


if __name__ == "__main__":
    main = Main()
    main.execute()
