import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union

import commentjson as cjson

user_settings_path = "./user_settings.json"


def factory(json_path: str = None) -> "JsonSchemaSettings":
    empty_jss = JsonSchemaSettings(dataset_path="./dataset")

    json_path = json_path if json_path is not None else user_settings_path
    if not os.path.exists(json_path):
        print(f"Don't load user toml. (The '{json_path}' is not exist.)\n")
        # TODO
        # ul.raise ... or return empty_jss
        return empty_jss
        # raise FileNotFoundError

    # load
    data = open_json(json_path)
    # delete key
    data.pop("$schema")

    dest_paths: Optional[dict] = data.pop("destinations", None)
    new_data = iter_dict(data)

    if dest_paths is not None:
        for k, v in dest_paths.items():
            k = camel2snake(k)
            new_data[f"{k}_path"] = v

    jss = JsonSchemaSettings(**new_data)
    return jss


def open_json(path: str) -> dict:
    with open(path) as f:
        # data = cjson.load(f, object_pairs_hook=OrderedDict)
        data = cjson.load(f)
    return data


def camel2snake(obj: str):
    return "".join([f"_{i.lower()}" if i.isupper() else i for i in obj]).lstrip("_")


def iter_dict(_dict: dict):
    new_dict = dict()

    for key, val in _dict.items():
        if isinstance(val, dict):
            iter_dict(val)
        else:
            x = camel2snake(key)
            new_dict[x] = val
            # make_self(x, val)
    return new_dict


# def make_self(k: str, v: str):
#     ss = "self.{} = {}"

#     has_backslash = False if not isinstance(v, str) else v.find("\\") > -1
#     if has_backslash:
#         ss = 'self.{} = r"{}"'

#     elif isinstance(v, str):
#         v.replace('"', "'")  # "" -> ''
#         ss = 'self.{} = "{}"'

#     expression = ss.format(k, v)
#     # exec(expression)
#     print(expression)


@dataclass
class JsonSchemaSettings:
    # cnn-with-pytorch/json_schema/settings_schema.json
    dataset_path: str

    height: int = 60
    width: int = 60
    limit_size_each_class: int = -1
    test_size: Union[int, float] = 0.1
    extensions: list = field(default_factory=list)
    channels: int = 3

    epoch: int = 10
    batch: int = 128
    subdivision: int = 4
    save_cycle: int = 0
    test_cycle: int = 1
    use_gpu: bool = True
    save_final_model: bool = True
    shuffle_dataset_per_epoch: bool = True

    false_path: str = "./false"
    trained_model_path: str = "./"
    config_path: str = "./config"
    log_path: str = "./logs"
    grad_cam_path: str = "./GradCAM_results"

    is_execute_grad_cam: bool = True
    grad_cam_layer: str = "conv5"
    is_execute_grad_cam_when_only_mistake: bool = True

    is_available_re_training: bool = False
    is_re_training_model: bool = False
    load_model_path: str = ""

    is_show_network_difinition: bool = True
    is_save_debug_log: bool = True
    is_save_rate_log: bool = True

    r"""
    ... = ["abc", "def"] のようにデフォルト値を持てない
        -> field(default_factory=list)で空のリストとして定義しておく
        -> __post_init__()で判定してデフォルト値をセット

    Traceback (most recent call last):
    File "test/test.py", line 46, in <module>
        @dataclass
    File "D:\scoop\kazuya\apps\python\3.7.4\lib\dataclasses.py", line 991, in dataclass
        return wrap(_cls)
    File "D:\scoop\kazuya\apps\python\3.7.4\lib\dataclasses.py", line 983, in wrap
        return _process_class(cls, init, repr, eq, order, unsafe_hash, frozen)
    File "D:\scoop\kazuya\apps\python\3.7.4\lib\dataclasses.py", line 834, in _process_class
        for name, type in cls_annotations.items()]
    File "D:\scoop\kazuya\apps\python\3.7.4\lib\dataclasses.py", line 834, in <listcomp>
        for name, type in cls_annotations.items()]
    File "D:\scoop\kazuya\apps\python\3.7.4\lib\dataclasses.py", line 727, in _get_field
        raise ValueError(f'mutable default {type(f.default)} for field '
    ValueError: mutable default <class 'list'> for field extensions is not allowed: use default_factory
    """

    def __post_init__(self):
        # list -> 中身があればTrueを返す / なければFalse
        if not self.extensions:
            self.extensions = ["jpg", "png", "jpeg"]

        # options
        # self.filename_base: str
        # self.input_size: Tuple[int, int]  # (height, width)

        # determine input size
        self.input_size = (self.height, self.width)
        self.filename_base = datetime.now().strftime("%Y%b%d_%Hh%Mm%Ss")


if __name__ == "__main__":
    from pprint import pprint as pp

    jss = factory()
    pp(jss.__dict__)

    r"""MEMO
    dict ⊂ OrderedDict

    >>> from collections import OrderedDict
    >>> od = OrderedDict({"k": 1})
    >>> d = dict({"k": 2})
    >>> od
    OrderedDict([('k', 1)])
    >>> d
    {'k': 2}
    >>> type(od)
    <class 'collections.OrderedDict'>
    >>> type(d)
    <class 'dict'>
    >>> isinstance(od, type(d))
    True
    >>> isinstance(d, type(od))
    False
    """
