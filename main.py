from modules.yaml_parser import GlobalConfig
import sys
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
import torchvision.transforms as transforms

# my packages
import modules.utils as ul
import modules.torch_utils as tu
from modules import factory_config

# import modules as m


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help="path of `user_settings.toml`.", default=None)

    return parser.parse_args()


def preprocess() -> GlobalConfig:
    args = parse_argument()
    yaml_path = args.path
    if yaml_path is not None:
        print(f"Set `yaml_path` to '{yaml_path}'")

    print(f"Reading config from '{yaml_path}'...")
    return factory_config(yaml_path)


def main() -> int:
    r"""Main process.

    Raises:
        error: FileNotFound.

    Returns:
        int: exit status.
    """
    # load config.
    GCONF = preprocess()

    """ DEBUG NOW """
    GCONF.option.is_save_debug_log = False
    GCONF.option.is_save_rate_log = False
    GCONF.option.is_save_mistaken_pred = False
    GCONF.option.is_save_used_image_path = False
    """ --------- """

    # # if not exist, raise error
    ul.raise_when_FileNotFound(GCONF.path.dataset)

    # ===== log / rate file =====
    GCONF.log = ul.LogFile(std_debug_ok=False)
    GCONF.rate = ul.LogFile(std_debug_ok=False)

    # debug log
    if GCONF.option.is_save_debug_log:
        p = ul.create_file_path(GCONF.path.log, GCONF.filename_base)
        GCONF.log = ul.LogFile(p, std_debug_ok=False)

    # rate log
    if GCONF.option.is_save_rate_log:
        p = ul.create_file_path(GCONF.path.log, GCONF.filename_base, ext="csv")
        GCONF.rate = ul.LogFile(p, std_debug_ok=False)

    # ===== datasets =====
    # progress = ul.ProgressLog(f"Create dataset from '{GCONF.path.dataset}'")
    print(f"Create dataset from '{GCONF.path.dataset}'...")
    # train, unknown, known
    dataset = tu.CreateDataset(GCONF=GCONF)

    # ===== calculate dataset normalization =====
    # progress = ul.ProgressLog('Calculating dataset normalization')

    # import time
    # st = time.time()
    # print(time.time() - st)
    # mean, std = tu.calc_dataset_norm(dataset, GCONF.network.channels)
    # print(f'mean: {mean}')
    # print(f'std : {std}')

    # progress.complete()

    # transform
    transform = transforms.Compose(
        [
            transforms.Resize(GCONF.network.input_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # loader = {'train': [...], 'unknown': [...], 'known': [...]}
    loader = dataset.create_dataloader(
        GCONF.network.batch, transform, GCONF.network.is_shuffle_dataset_per_epoch
    )

    # make directories
    pth_save_path = GCONF.path.model if GCONF.network.is_save_final_model else None

    is_save_log = GCONF.option.is_save_debug_log or GCONF.option.is_save_rate_log
    log_path = GCONF.path.log if is_save_log else None

    ul.make_directories(
        GCONF.path.mistaken, log_path, pth_save_path,
    )

    # rate log
    cls_list = dataset.classes.values()
    _ = ", " * len(cls_list)
    GCONF.rate.writeline(f", unknown {_}, known {_}")
    _ = ", ".join(cls_list)
    GCONF.rate.writeline(f"Test No, {_}, TOTAL, {_}, TOTAL")

    # ===== network =====
    # progress = ul.ProgressLog("Building CNN network")  # debug log
    print("Building CNN network...")
    # build network
    model = tu.Model(classes=dataset.classes, loader=loader, GCONF=GCONF)

    # logging parameter config
    if GCONF.option.is_show_network_difinition:
        _show_network_difinition(model, dataset)

    # ===== training model =====
    model.train()  # train
    model.test()  # final test

    # ===== save model =====
    # final pth
    if GCONF.network.is_save_final_model:
        # pth path
        save_path = ul.create_file_path(
            GCONF.path.model, GCONF.filename_base, end="_final", ext="pth"
        )

        # progress = ul.ProgressLog(f"Saving model to '{save_path}'")
        print(f"Saving model to '{save_path}'...")

        save_option = [True, True] if GCONF.option.is_available_re_training else [False, False]
        model.save(save_path, *save_option)  # save

        GCONF.log.writeline(f"# Saved model to '{save_path}'", debug_ok=False)

    # close file when opening.
    GCONF.log.close()
    GCONF.rate.close()

    return 0


def _show_network_difinition(model: tu.Model, dataset: tu.CreateDataset) -> None:
    r"""Show network difinition on console.

    Args:
        model (tu.Model): model.
        dataset (tu.CreateDataset): dataset.
    """
    GCONF = model.GCONF

    global_conf = {
        "run time": GCONF.filename_base,
        "image path": GCONF.path.dataset,
        "supported extensions": GCONF.dataset.extensions,
        "saving debug log is": GCONF.option.is_save_debug_log,
        "saving rate log is": GCONF.option.is_save_rate_log,
        "pth save cycle": GCONF.network.save_cycle,
        "test cycle": GCONF.network.test_cycle,
        "saving final pth is": GCONF.network.is_save_final_model,
        "Grad-CAM is": GCONF.gradcam.enabled,
        "Grad-CAM layer": GCONF.gradcam.layer,
        "load model is ": GCONF.option.load_model_path,
        "load path": GCONF.option.load_model_path if GCONF.option.re_training else "None",
    }

    dataset_conf = {
        "limit dataset size": GCONF.dataset.limit_size,
        "train dataset size": dataset.train_size,
        "unknown dataset size": dataset.unknown_size,
        "known dataset size": dataset.known_size,
    }

    model_conf = {
        "net": str(model.net),
        "optimizer": str(model.optimizer),
        "criterion": str(model.criterion),
        "input size": f"(h: {GCONF.network.height}, w: {GCONF.network.width})",
        "epoch": GCONF.network.epoch,
        "batch size": GCONF.network.batch,
        "subdivision": GCONF.network.subdivision,
        "GPU available": torch.cuda.is_available(),
        "GPU used": GCONF.network.gpu_enabled and torch.cuda.is_available(),
        "re-training": ("available" if GCONF.option.is_available_re_training else "not available"),
    }

    def _inner_execute(_dict: Dict[str, Any], head: str = "") -> None:
        r"""execute.

        Args:
            _dict (Dict[str, Any]): show contents.
            head (str, optional): show before showing contents. Defaults to ''.
        """
        GCONF.log.writeline(head, debug_ok=True)

        # adjust to max length of key
        max_len = max([len(x) for x in _dict.keys()])
        # _format = f'%-{max_len}s : %s'

        for k, v in _dict.items():
            # format for structure of network
            if isinstance(v, str) and v.find("\n") > -1:
                v = v.replace("\n", "\n" + " " * (max_len + 3)).rstrip()

            GCONF.log.writeline(f"{k.center(max_len)} : {v}", debug_ok=True)
        GCONF.log.writeline("", debug_ok=True)

    classes = {str(k): v for k, v in model.classes.items()}

    print()
    _inner_execute(classes, "--- Classes ---")
    _inner_execute(global_conf, "--- Global Configuration ---")
    _inner_execute(dataset_conf, "--- Dataset Configuration ---")
    _inner_execute(model_conf, "--- Model Configuration ---")


if __name__ == "__main__":
    sys.exit(main())
