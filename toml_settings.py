from typing import List, Optional, Tuple, Union
import toml


def apply_settings(target: dict, src: dict):
    """ Apply key and value of src to target dictionary.
            target <- src
    """

    keys: str
    vals: dict

    # overwrite
    for keys, vals in src.items():
        for k, v in vals.items():
            # if key is not exist
            if not(keys in target and k in target[keys]):
                continue

            target[keys][k] = v


def print_toml(_dict: dict):
    keys: str
    vals: dict

    for keys, vals in _dict.items():
        print(f'[{keys}]')
        for k, v in vals.items():
            print(f'  {k}: {v}')
        print()


def print_dict(_dict: dict):
    for k, v in _dict.items():
        print(f'{k}: {v}')


def factory() -> 'TomlSettings':
    pre_toml = toml.load('./default_settings.toml')
    usr_toml = toml.load('./user_settings.toml')
    apply_settings(pre_toml, usr_toml)

    applied_toml = pre_toml.copy()

    tms = TomlSettings(applied_toml)
    return tms


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
        self.input_size: Tuple[int, int]
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

        from datetime import datetime
        self.filename_base = datetime.now().strftime('%Y%b%d_%Hh%Mm%Ss')

        # -----

        # dynamically definition by exec()
        #   `exec` reference <https://docs.python.org/3/library/functions.html#exec>

        # [Examples]
        #   self.dataset_path = r'./Images/'
        #   self.use_gpu = True
        #   self.epoch = 10

        for _, vals, in _dict.items():
            for k, v in vals.items():
                ss = 'self.{} = {}'

                if k.find('path') > -1 or k.find('\\') > -1:
                    ss = 'self.{} = r"{}"'

                elif isinstance(v, str):
                    v.replace('"', '\'')  # "" -> ''
                    ss = 'self.{} = "{}"'

                expression = ss.format(k, v)
                exec(expression)

        # determine input size
        if isinstance(self.input_size, list):
            self.input_size = tuple(self.input_size)

        if self.limit_dataset_size == 0:
            self.limit_dataset_size = None
