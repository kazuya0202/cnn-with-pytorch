from dataclasses import dataclass
from typing import Any, Dict

import torch
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

        # # if not exist, raise error
        ul.raise_when_FileNotFound(tms.dataset_path)

        # ===== log / rate file =====
        log = ul.LogFile(None)
        rate = ul.LogFile(None)

        # debug log
        if tms.is_save_debug_log:
            p = ul.create_file_path(tms.log_path, tms.filename_base)
            log = ul.LogFile(p, std_debug_ok=True)

        # rate log
        if tms.is_save_rate_log:
            p = ul.create_file_path(tms.log_path, tms.filename_base, ext='csv')
            rate = ul.LogFile(p, std_debug_ok=True)

        # ===== datasets =====
        progress = ul.ProgressLog(f'Create dataset from \'{tms.dataset_path}\'')

        # transform
        transform = transforms.Compose([
            transforms.Resize(tms.input_size),
            transforms.ToTensor()])

        # train, unknown, known
        dataset = tu.CreateDataset(
            path=tms.dataset_path,
            extensions=tms.extensions,
            test_size=tms.test_size,
            config_path=tms.config_path)

        train_data, unknown_data, known_data = self._create_custom_dataloader(
            dataset=dataset,
            batch_size=tms.batch,
            transform=transform,
            is_shuffle=tms.is_shuffle_per_epoch)

        progress.complete()

        # mkdir
        pth_save_path = tms.pth_save_path if tms.is_save_final_pth else None

        is_save_log = tms.is_save_debug_log or tms.is_save_rate_log
        log_path = tms.log_path if is_save_log else None

        ul.make_directories(
            tms.false_path,
            log_path,
            pth_save_path,
        )

        # ===== network =====
        progress = ul.ProgressLog('Building CNN network')  # debug log

        # build network
        model = tu.Model(
            toml_settings=tms,
            classes=dataset.classes,
            use_gpu=tms.use_gpu,
            log=log,
            rate=rate)

        progress.complete()

        # logging parameter config
        if tms.is_print_network_difinition:
            self._print_network_difinition(model, dataset, log)

        # ===== dataset model =====
        test_model = tu.TestModel(model, unknown_data, known_data)
        train_model = tu.TrainModel(model, tms.epoch, train_data, test_model)

        # exec
        train_model.train()  # train
        test_model.test(epoch=tms.epoch - 1)  # final test

        # ===== save model =====
        # pth path
        save_path = ul.create_file_path(
            tms.pth_save_path, tms.filename_base, end='_final', ext='pth')

        # final pth
        if tms.is_save_final_pth:
            progress = ul.ProgressLog(f'Saving model to \'{save_path}\'')
            train_model.save_model(save_path)  # save
            progress.complete()

            log.writeline(f'# Saved model to \'{save_path}\'', debug_ok=False)

        if tms.is_save_debug_log:
            log.close()
        if tms.is_save_rate_log:
            rate.close()
    # end of [function] execute

    def _create_custom_dataloader(
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
    # end of [function] _create_custom_dataset

    def _print_network_difinition(
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
            'train dataset size': dataset.train_size,
            'unknown dataset size': dataset.unknown_size,
            'known dataset size': dataset.known_size,
        }

        model_conf = {
            'net': str(model.net),
            'optimizer': str(model.optimizer),
            'criterion': str(model.criterion),
            'input size': f'(h: {tms.height}, w: {tms.width})',
            'epoch': tms.epoch,
            'subdivision': tms.subdivision,
            'GPU available': torch.cuda.is_available(),
            'GPU used': model.use_gpu
        }

        def _inner_execute(_dict: Dict[str, Any], head: str = ''):
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

        _inner_execute(classes, '--- Classify Classes ---')
        _inner_execute(global_conf, '--- Global Config ---')
        _inner_execute(dataset_conf, '--- Dataset Config ---')
        _inner_execute(model_conf, '--- Model Config ---')
    # end of [function] _print_parameter_config
# end of [class] Main


if __name__ == "__main__":
    main = Main()
    main.execute()
