import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, Union

import toml

# default path
user_settings_path = "./user_settings.toml"


def factory(toml_path: str = None) -> "TomlSettings":
    tms = TomlSettings()

    toml_path = toml_path if toml_path is not None else user_settings_path

    if not os.path.exists(toml_path):
        print(f"Don't load user toml. (The '{toml_path}' is not exist.)\n")
        return tms

    toml_ = toml.load(toml_path)  # load
    tms.apply_settings(toml_)
    return tms


@dataclass(init=False)
class TomlSettings:
    def __init__(self):
        # default settings
        #   for autocomplete and explicitly definition

        # [common]
        # false - save mistaking images.
        self.false_path: str = r"./Recognition/False/"
        # path - save trained model.
        self.pth_save_path: str = r"./Recognition/"
        # config - save config of dataset.
        self.config_path: str = r"./config/"
        # load pth model.
        self.is_load_model: bool = False
        # path - load model. (only `is_load_model` is True.)
        self.load_pth_path: str = r""

        # [dataset]
        # image - target dataset.
        self.dataset_path: str = r"./Recognition/Images/"
        # max dataset size.
        self.limit_dataset_size: Optional[int] = 0
        # testing size.
        self.test_size: Union[float, int] = 0.1  # rate
        #   self.test_size: Union[float, int] = 400  #number
        # supported extensions.
        self.extensions: List[str] = ["jpg", "png", "jpeg", "bmp", "gif"]
        # chanenls of dataset images.
        self.channels: int = 3

        # [network]
        # input size to network.
        self.height: int = 60
        self.width: int = 60
        # epoch times.
        self.epoch: int = 10
        # image size in one batch.
        self.batch: int = 128
        # subdivision of batch.
        self.subdivision: int = 4
        # testing cycle.
        self.test_cycle: int = 1
        # saving pth cycle.
        self.pth_save_cycle: int = 0
        # using gpu.
        self.use_gpu: bool = True
        # shuffle images per epoch.
        self.is_shuffle_per_epoch: bool = True
        # save final pth.
        self.is_save_final_pth: bool = True
        # is available re-training model.
        self.is_available_re_training: bool = False

        # [log]
        # log - save logging.
        self.log_path: str = r"./Recognition/Logs/"
        # debug log.
        self.is_save_debug_log: bool = True
        # rate log.
        self.is_save_rate_log: bool = True
        # network definition.
        self.is_show_network_difinition: bool = True

        # [gradcam]
        # save path of grad cam images.
        self.grad_cam_path: str = r"./Recognition/grad_cam_results/"
        # use grad cam.
        self.is_grad_cam: bool = True
        # visualize layer.
        self.grad_cam_layer: str = "conv5"

        # options
        self.filename_base: str
        self.input_size: Tuple[int, int]  # (height, width)

    def apply_settings(self, _dict: dict):
        # dynamically definition by exec()
        #   `exec` reference <https://docs.python.org/3/library/functions.html#exec>

        # [Examples]
        #   self.dataset_path = r'.\Images'
        #   self.use_gpu = True
        #   self.epoch = 10

        for _, vals, in _dict.items():
            for k, v in vals.items():
                ss = "self.{} = {}"

                has_backslash = False if not isinstance(v, str) else v.find("\\") > -1

                # if k.find('path') > -1 or v.find('\\') > -1:
                if k.find("path") > -1 or has_backslash:
                    ss = 'self.{} = r"{}"'

                elif isinstance(v, str):
                    v.replace('"', "'")  # "" -> ''
                    ss = 'self.{} = "{}"'

                expression = ss.format(k, v)
                exec(expression)

        if self.limit_dataset_size == -1:
            self.limit_dataset_size = None

        # determine input size
        self.input_size = (self.height, self.width)
        self.filename_base = datetime.now().strftime("%Y%b%d_%Hh%Mm%Ss")
