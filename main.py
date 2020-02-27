import sys
from typing import Any, Dict

import torch
import torchvision.transforms as transforms

# my packages
import toml_settings as _tms
import torch_utils as tu
import utils as ul


def main() -> int:
    r"""Main process.

    Raises:
        error: FileNotFound.

    Returns:
        int: exit status.
    """
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

    # loader = {'train': [...], 'unknown': [...], 'known': [...]}
    loader = dataset.create_dataloader(
        tms.batch, transform, tms.is_shuffle_per_epoch)

    progress.complete()

    # make directories
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
        classes=dataset.classes,
        loader=loader,
        tms=tms,
        use_gpu=tms.use_gpu,
        log=log,  # **options
        rate=rate,  # **options
    )

    progress.complete()

    # logging parameter config
    if tms.is_show_network_difinition:
        _show_network_difinition(model, dataset, log)

    # ===== training model =====
    model.train()  # train
    model.test()  # final test

    # ===== save model =====
    # pth path
    save_path = ul.create_file_path(
        tms.pth_save_path, tms.filename_base, end='_final', ext='pth')

    # final pth
    if tms.is_save_final_pth:
        progress = ul.ProgressLog(f'Saving model to \'{save_path}\'')
        model.save_model(save_path)  # save
        progress.complete()

        log.writeline(f'# Saved model to \'{save_path}\'', debug_ok=False)

    # close file when opening.
    if tms.is_save_debug_log:
        log.close()
    if tms.is_save_rate_log:
        rate.close()

    return 0
# end of [function] main


def _show_network_difinition(model: tu.Model, dataset: tu.CreateDataset,
                             log: ul.LogFile) -> None:
    r"""Show network difinition on console.

    Args:
        model (tu.Model): model.
        dataset (tu.CreateDataset): dataset.
        log (ul.LogFile): log.
    """
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
        'GPU used': tms.use_gpu and torch.cuda.is_available(),
    }

    def _inner_execute(_dict: Dict[str, Any], head: str = '') -> None:
        r"""execute.

        Args:
            _dict (Dict[str, Any]): show contents.
            head (str, optional): show before showing contents. Defaults to ''.
        """
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
    # end of [function] _inner_execute

    classes = {str(k): v for k, v in model.classes.items()}

    _inner_execute(classes, '--- Classify Classes ---')
    _inner_execute(global_conf, '--- Global Config ---')
    _inner_execute(dataset_conf, '--- Dataset Config ---')
    _inner_execute(model_conf, '--- Model Config ---')
# end of [function] _show_parameter_config


if __name__ == "__main__":
    sys.exit(main())
