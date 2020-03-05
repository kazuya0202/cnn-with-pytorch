from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import toml
import os

# my packages
import utils as ul

user_settings_path = './user_settings.toml'


def factory(toml_path=None) -> 'TomlSettings':
    tms = TomlSettings()

    toml_path = toml_path if toml_path is not None \
        else user_settings_path

    if not os.path.exists(toml_path):
        print(f'Don\'t load user toml.(the \'{toml_path}\' is not exist.\n)')
        return tms

    toml_ = toml.load(toml_path)  # load
    tms.apply_settings(toml_)
    return tms
# end of [function] factory


@dataclass(init=False)
class TomlSettings:
    def __init__(self):
        # default settings
        #   for autocomplete and explicitly definition

        # [common]
        # false - save mistaking images.
        self.false_path: str = './Recognition/False/'
        # path - save trained model.
        self.pth_save_path: str = './Recognition/'
        # config - save config of dataset.
        self.config_path: str = './config/'
        # load pth model.
        self.is_load_model: bool = False
        # path - load model. (only `is_load_model` is True.)
        self.load_pth_path: str = ''

        # [dataset]
        # image - target dataset.
        self.dataset_path: str = './Recognition/Images/'
        # max dataset size.
        self.limit_dataset_size: Optional[int] = 0
        # testing size.
        self.test_size: Union[float, int] = 0.1  # rate
        #   self.test_size: Union[float, int] = 400  #number
        # supported extensions.
        self.extensions: List[str] = ['jpg', 'png', 'jpeg', 'bmp', 'gif']
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
        self.log_path: str = './Recognition/Logs/'
        # debug log.
        self.is_save_debug_log: bool = True
        # rate log.
        self.is_save_rate_log: bool = True
        # network definition.
        self.is_show_network_difinition: bool = True

        # [gradcam]
        # save path of grad cam images.
        self.grad_cam_path: str = './Recognition/grad_cam_results/'
        # use grad cam.
        self.is_grad_cam: bool = True
        # visualize layer.
        self.grad_cam_layer: str = 'conv5'

        # options
        self.filename_base: str
        self.input_size: Tuple[int, int]

    def apply_settings(self, _dict: dict):
        # dynamically definition by exec()
        #   `exec` reference <https://docs.python.org/3/library/functions.html#exec>

        # [Examples]
        #   self.dataset_path = r'.\Images'
        #   self.use_gpu = True
        #   self.epoch = 10

        for _, vals, in _dict.items():
            for k, v in vals.items():
                ss = 'self.{} = {}'

                has_backslash = False if not isinstance(v, str) \
                    else v.find('\\') > -1

                # if k.find('path') > -1 or k.find('\\') > -1:
                if ul.find_str(k, 'path') or has_backslash:
                    ss = 'self.{} = r"{}"'

                elif isinstance(v, str):
                    v.replace('"', '\'')  # "" -> ''
                    ss = 'self.{} = "{}"'

                expression = ss.format(k, v)
                exec(expression)

        if self.limit_dataset_size == -1:
            self.limit_dataset_size = None

        # determine input size
        self.input_size = (self.height, self.width)

        from datetime import datetime
        self.filename_base = datetime.now().strftime('%Y%b%d_%Hh%Mm%Ss')

    # end of [function] __init__
# end of [class] TomlSettings

# ====================================================== #

# paths = {
#     'default': './default_settings.toml',
#     'user': './user_settings.toml'
# }


# def apply_settings(target: dict, src: dict):
#     """ Apply key and value of src to target dictionary.
#             target <- src
#     """

#     key: str
#     vals: dict

#     # overwrite
#     for key, vals in src.items():
#         for k, v in vals.items():
#             # if key is not exist
#             if not(key in target and k in target[key]):
#                 continue

#             target[key][k] = v
# # end of [function] apply_settings


# def print_toml(_dict: dict):
#     keys: str
#     vals: dict

#     for keys, vals in _dict.items():
#         print(f'[{keys}]')
#         for k, v in vals.items():
#             print(f'  {k}: {v}')
#         print()
# # end of [function] print_toml


# def print_dict(_dict: dict):
#     for k, v in _dict.items():
#         print(f'{k}: {v}')
# # end of [function] print_dict


# def factory(default_path: Optional[str] = None,
#             user_path: Optional[str] = None) -> 'TomlSettings':
#     default_path = default_path if default_path is not None \
#         else paths['default']
#     user_path = user_path if user_path is not None \
#         else paths['user']

#     # raise
#     ul.raise_when_FileNotFound(default_path)
#     ul.raise_when_FileNotFound(user_path)

#     pre_toml = toml.load(default_path)
#     usr_toml = toml.load(user_path)

#     apply_settings(pre_toml, usr_toml)

#     applied_toml = pre_toml.copy()
#     tms = TomlSettings(applied_toml)
#     return tms
# # end of [function] factory
