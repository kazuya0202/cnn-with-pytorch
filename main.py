import errno
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# my packages
import toml_settings as _tms
import torch_utils as tu
import utils as ul


@dataclass
class Main:
    def execute(self):
        tms = _tms.factory()

        """ DEBUG NOW """
        tms.is_save_debug_log = False
        tms.is_save_rate_log = False
        """ --------- """

        # # if not exist, exit script
        if not Path(tms.dataset_path).exists():
            err = OSError(errno.ENOENT, os.strerror(errno.ENOENT), tms.dataset_path)
            raise FileNotFoundError(err)

        # ===== log / rate file =====
        log = ul.LogFile(None)
        rate = ul.LogFile(None)

        # if gv.is_save_debug_log:
        if tms.is_save_debug_log:
            p = ul.create_file_path(tms.log_path, tms.filename_base)
            log = ul.LogFile(p, default_debug_ok=True)

        # if gv.is_save_rate_log:
        if tms.is_save_rate_log:
            p = ul.create_file_path(tms.log_path, tms.filename_base, ext='csv')
            rate = ul.LogFile(p, default_debug_ok=True)

        # ===== create datasets =====
        progress = ul.ProgressLog(f'Create dataset from \'{tms.dataset_path}\'')

        # transform
        transform = transforms.Compose([
            transforms.Resize(tms.input_size),
            transforms.ToTensor()])

        # train, test(unknown), test(known)
        dataset = tu.CreateDataset(
            path=tms.dataset_path,
            extensions=tms.extensions,
            test_size=tms.test_size,
            config_path=tms.config_path)

        train_data, unknown_data, known_data = self.__create_custom_dataloader(
            dataset=dataset,
            batch_size=tms.batch,
            transform=transform,
            is_shuffle=tms.is_shuffle_per_epoch)

        # train_data, unknown_data = self.__create_mnist_dataloader(
        #     # dataset=dataset,
        #     batch_size=gv.batch_size,
        #     transform=transform,
        #     is_shuffle=gv.is_shuffle_per_epoch)
        # known_data = None

        progress.complete()

        # ===== create make required direcotry =====
        progress = ul.ProgressLog('Making required directory')

        # make required path
        ul.make_directories(
            tms.false_path,
            tms.log_path,
            tms.pth_save_path
        )
        progress.complete()

        # ===== network =====
        progress = ul.ProgressLog('Building CNN network')  # debug log
        # create network
        model = tu.Model(
            classes=dataset.classes,
            use_gpu=tms.use_gpu,
            log=log,
            rate=rate,
            toml_settings=tms)

        progress.complete()

        # logging parameter config
        if tms.is_print_network_difinition:
            self.__print_parameter_config(model, dataset, log)

        # ===== dataset model =====
        test_model = tu.TestModel(model, unknown_data, known_data)

        train_model = tu.TrainModel(
            model=model,
            epoch=tms.epoch,
            train_data=train_data,
            test_model=test_model)

        # exec
        train_model.train()  # train
        test_model.test(epoch=tms.epoch - 1)  # final test

        # ===== save model =====
        # pth path
        pt_params = [tms.pth_save_path, tms.filename_base]
        save_path = ul.create_file_path(*pt_params, end='_final', ext='pth')

        progress = ul.ProgressLog(f'Saving model to \'{save_path}\'')
        if tms.is_save_final_pth:
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

        # train dataloader / test dataloader, shuffle
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=is_shuffle)
        unknown_data = DataLoader(unknown_dataset, batch_size=1, shuffle=is_shuffle)
        known_data = DataLoader(known_dataset, batch_size=1, shuffle=is_shuffle)

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
            model: tu.Model,
            dataset: tu.CreateDataset,
            log: ul.LogFile) -> None:

        tms = model.tms

        global_conf = {
            'run time': tms.filename_base,
            'image path': tms.dataset_path,
            'supported extensions': tms.extensions,
            'saving debug log is': tms.is_save_debug_log,
            'saving rate log is': tms.is_save_rate_log,
            'pth save cycle': tms.pth_save_cycle,
            'test cycle': tms.test_cycle,
            'saving final pth is': tms.is_save_final_pth,
            'Grad-CAM is': tms.is_grad_cam,
            'Grad-CAM layer': tms.grad_cam_layer
        }

        dataset_conf = {
            'limit dataset size': tms.limit_dataset_size,
            'train dataset size': len(dataset.all_list['train']),
            'unknown dataset size': len(dataset.all_list['unknown']),
            'known dataset size': len(dataset.all_list['known']),
        }

        model_conf = {
            'net': str(model.net),
            'optimizer': str(model.optimizer),
            'criterion': str(model.criterion),
            'input size': model.input_size,
            'epoch': tms.epoch,
            'subdivision': tms.subdivision,
            'GPU available': torch.cuda.is_available(),
            'GPU used': model.use_gpu
        }

        def __inner_execute(_dict: Dict[str, Any], head: str = ''):
            log.writeline(head)

            # adjust to max length of key
            max_len = max([len(x) for x in _dict.keys()])
            # _format = f'%-{max_len}s : %s'

            for k, v in _dict.items():
                # format for structure of network
                if isinstance(v, str) and v.find('\n') > -1:
                    v = v.replace('\n', '\n' + ' ' * (max_len + 3)).rstrip()

                log.writeline(f'{k.center(max_len)} : {v}')
            log.writeline('\n')

        classes = {str(k): v for k, v in model.classes.items()}

        __inner_execute(classes, '--- Classify Classes ---')
        __inner_execute(global_conf, '--- Global Config ---')
        __inner_execute(dataset_conf, '--- Dataset Config ---')
        __inner_execute(model_conf, '--- Model Config ---')
    # end of [function] __print_parameter_config
# end of [class] Main


if __name__ == "__main__":
    main = Main()
    main.execute()
