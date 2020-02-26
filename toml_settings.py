from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import toml

# my packages
import utils as ul

paths = {
    'default': './default_settings.toml',
    'user': './user_settings.toml'
}


def apply_settings(target: dict, src: dict):
    """ Apply key and value of src to target dictionary.
            target <- src
    """

    key: str
    vals: dict

    # overwrite
    for key, vals in src.items():
        for k, v in vals.items():
            # if key is not exist
            if not(key in target and k in target[key]):
                continue

            target[key][k] = v
# end of [function] apply_settings


def print_toml(_dict: dict):
    keys: str
    vals: dict

    for keys, vals in _dict.items():
        print(f'[{keys}]')
        for k, v in vals.items():
            print(f'  {k}: {v}')
        print()
# end of [function] print_toml


def print_dict(_dict: dict):
    for k, v in _dict.items():
        print(f'{k}: {v}')
# end of [function] print_dict


def factory(
        default_path: Optional[str] = None,
        user_path: Optional[str] = None) -> 'TomlSettings':

    global paths

    default_path = default_path if default_path is not None \
        else paths['default']
    user_path = user_path if user_path is not None \
        else paths['user']

    # raise
    ul.raise_when_FileNotFound(default_path)
    ul.raise_when_FileNotFound(user_path)

    pre_toml = toml.load(default_path)
    usr_toml = toml.load(user_path)

    apply_settings(pre_toml, usr_toml)

    applied_toml = pre_toml.copy()

    tms = TomlSettings(applied_toml)
    return tms
# end of [function] factory


@dataclass(init=False)
class TomlSettings:
    def __init__(self, _dict: dict):
        # default_settings.toml
        #   for autocomplete and explicitly definition

        # [util_path]
        self.false_path: str
        self.pth_save_path: str
        self.config_path: str

        # [dataset]
        self.dataset_path: str
        self.limit_dataset_size: Optional[int]
        self.test_size: Union[float, int]
        self.extensions: List[str]

        # [network]
        self.height: int
        self.width: int
        self.epoch: int
        self.batch: int
        self.subdivision: int
        self.use_gpu: bool
        self.test_cycle: int
        self.pth_save_cycle: int
        self.is_shuffle_per_epoch: bool
        self.is_save_final_pth: bool

        # [log]
        self.log_path: str
        self.is_save_debug_log: bool
        self.is_save_rate_log: bool
        self.is_print_network_difinition: bool

        # [gradcam]
        self.is_grad_cam: bool
        self.grad_cam_path: str
        self.grad_cam_layer: str

        # options
        self.filename_base: str
        self.input_size: Tuple[int, int]

        # -----

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

        if self.limit_dataset_size == 0:
            self.limit_dataset_size = None

        # determine input size
        self.input_size = (self.height, self.width)

        from datetime import datetime
        self.filename_base = datetime.now().strftime('%Y%b%d_%Hh%Mm%Ss')

    # end of [function] __init__
# end of [class] TomlSettings
