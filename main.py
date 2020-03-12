import sys
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
import torchvision.transforms as transforms

# my packages
import modules.toml_settings as _tms
import modules.utils as ul
import modules.torch_utils as tu


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help="path of `user_settings.toml`.", default=None)

    return parser.parse_args()


def main() -> int:
    r"""Main process.

    Raises:
        error: FileNotFound.

    Returns:
        int: exit status.
    """

    # determine toml path
    args = parse_argument()
    toml_path = args.path
    if toml_path is not None:
        print(f"Set `toml_path` to {toml_path}")

    # factory toml settings
    tms = _tms.factory(toml_path)

    """ DEBUG NOW """
    tms.is_save_debug_log = False
    tms.is_save_rate_log = False
    """ --------- """

    # # if not exist, raise error
    ul.raise_when_FileNotFound(tms.dataset_path)

    # ===== log / rate file =====
    log = ul.LogFile(std_debug_ok=False)
    rate = ul.LogFile(std_debug_ok=False)

    # debug log
    if tms.is_save_debug_log:
        p = ul.create_file_path(tms.log_path, tms.filename_base)
        log = ul.LogFile(p, std_debug_ok=False)

    # rate log
    if tms.is_save_rate_log:
        p = ul.create_file_path(tms.log_path, tms.filename_base, ext="csv")
        rate = ul.LogFile(p, std_debug_ok=False)

    # ===== datasets =====
    progress = ul.ProgressLog(f"Create dataset from '{tms.dataset_path}'")

    # train, unknown, known
    dataset = tu.CreateDataset(
        path=tms.dataset_path,
        extensions=tms.extensions,
        test_size=tms.test_size,
        config_path=tms.config_path,
        limit_size=tms.limit_dataset_size,
    )
    progress.complete()

    # ===== calculate dataset normalization =====
    # progress = ul.ProgressLog('Calculating dataset normalization')

    # import time
    # st = time.time()
    # print(time.time() - st)
    # mean, std = tu.calc_dataset_norm(dataset, tms.channels)
    # print(f'mean: {mean}')
    # print(f'std : {std}')

    # progress.complete()

    # transform
    transform = transforms.Compose(
        [
            transforms.Resize(tms.input_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # loader = {'train': [...], 'unknown': [...], 'known': [...]}
    loader = dataset.create_dataloader(tms.batch, transform, tms.is_shuffle_per_epoch)

    # make directories
    pth_save_path = tms.pth_save_path if tms.is_save_final_pth else None

    is_save_log = tms.is_save_debug_log or tms.is_save_rate_log
    log_path = tms.log_path if is_save_log else None

    ul.make_directories(
        tms.false_path, log_path, pth_save_path,
    )

    # rate log
    cls_list = dataset.classes.values()
    _ = ", " * len(cls_list)
    rate.writeline(f", unknown {_}, known {_}")
    _ = ", ".join(cls_list)
    rate.writeline(f"Test No, {_}, TOTAL, {_}, TOTAL")

    # ===== network =====
    progress = ul.ProgressLog("Building CNN network")  # debug log

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
    # final pth
    if tms.is_save_final_pth:
        # pth path
        save_path = ul.create_file_path(
            tms.pth_save_path, tms.filename_base, end="_final", ext="pth"
        )

        progress = ul.ProgressLog(f"Saving model to '{save_path}'")

        save_option = [True, True] if tms.is_available_re_training else [False, False]
        model.save(save_path, *save_option)  # save

        progress.complete()

        log.writeline(f"# Saved model to '{save_path}'", debug_ok=False)

    # close file when opening.
    log.close()
    rate.close()

    return 0


def _show_network_difinition(model: tu.Model, dataset: tu.CreateDataset, log: ul.LogFile) -> None:
    r"""Show network difinition on console.

    Args:
        model (tu.Model): model.
        dataset (tu.CreateDataset): dataset.
        log (ul.LogFile): log.
    """
    tms = model.tms

    global_conf = {
        "run time": tms.filename_base,
        "image path": tms.dataset_path,
        "supported extensions": tms.extensions,
        "saving debug log is": tms.is_save_debug_log,
        "saving rate log is": tms.is_save_rate_log,
        "pth save cycle": tms.pth_save_cycle,
        "test cycle": tms.test_cycle,
        "saving final pth is": tms.is_save_final_pth,
        "Grad-CAM is": tms.is_grad_cam,
        "Grad-CAM layer": tms.grad_cam_layer,
        "load model is ": tms.is_load_model,
        "load path": tms.load_pth_path if tms.is_load_model else "None",
    }

    dataset_conf = {
        "limit dataset size": tms.limit_dataset_size,
        "train dataset size": dataset.train_size,
        "unknown dataset size": dataset.unknown_size,
        "known dataset size": dataset.known_size,
    }

    model_conf = {
        "net": str(model.net),
        "optimizer": str(model.optimizer),
        "criterion": str(model.criterion),
        "input size": f"(h: {tms.height}, w: {tms.width})",
        "epoch": tms.epoch,
        "batch size": tms.batch,
        "subdivision": tms.subdivision,
        "GPU available": torch.cuda.is_available(),
        "GPU used": tms.use_gpu and torch.cuda.is_available(),
        "re-training": ("available" if tms.is_available_re_training else "not available"),
    }

    def _inner_execute(_dict: Dict[str, Any], head: str = "") -> None:
        r"""execute.

        Args:
            _dict (Dict[str, Any]): show contents.
            head (str, optional): show before showing contents. Defaults to ''.
        """
        log.writeline(head, debug_ok=True)

        # adjust to max length of key
        max_len = max([len(x) for x in _dict.keys()])
        # _format = f'%-{max_len}s : %s'

        for k, v in _dict.items():
            # format for structure of network
            if isinstance(v, str) and v.find("\n") > -1:
                v = v.replace("\n", "\n" + " " * (max_len + 3)).rstrip()

            log.writeline(f"{k.center(max_len)} : {v}", debug_ok=True)
        log.writeline("\n", debug_ok=True)

    classes = {str(k): v for k, v in model.classes.items()}

    _inner_execute(classes, "--- Classify Classes ---")
    _inner_execute(global_conf, "--- Global Config ---")
    _inner_execute(dataset_conf, "--- Dataset Config ---")
    _inner_execute(model_conf, "--- Model Config ---")


def _get_rate_log_as_table():
    pass
    # rate log
    # ss = ul.set_align_center('Test No') + ' | '
    # ss += ul.set_align_center('unknown [all]', align=15) + ' | '
    # ss += ul.set_align_center('known [all]', align=15) + ' | '

    # cls_list = dataset.classes.values()
    # max_len = max([len(x) for x in cls_list])
    # cls_align = max(max_len, 10)

    # ss += ' | '.join([ul.set_align_center(x, cls_align) for x in cls_list])

    # import re
    # xs = list('\n' + ('-' * len(ss)))  # to character list
    # indexes = [x.start() for x in re.finditer(r'\|', ss)]
    # for x in indexes:
    #     xs[x + 1] = '+'

    # ss += ''.join(xs)  # to string and assign
    # rate.writeline(ss)
    # exit()


if __name__ == "__main__":
    sys.exit(main())
