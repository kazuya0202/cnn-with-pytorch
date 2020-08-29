from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union
import yaml
from dataclasses import dataclass, field


@dataclass
class Path_:
    dataset: str
    mistaken: str
    model: str
    config: str
    log: str
    gradcam: str


@dataclass
class Dataset_:
    limit_size: int
    test_size: Union[int, float]
    extensions: List[str]


@dataclass
class Gradcam_:
    enabled: bool
    only_mistaken: bool
    layer: str


@dataclass
class Network_:
    height: int
    width: int
    channels: int

    epoch: int
    batch: int
    subdivision: int

    save_cycle: int
    test_cycle: int

    gpu_enabled: bool
    save_final_model: bool
    shuffle_dataset_per_epoch: bool

    input_size: Tuple[int, int] = field(init=False) # height, width

    def __post_init__(self) -> None:
        self.input_size = (self.height, self.width)


@dataclass
class Option_:
    is_show_network_difinition: bool
    is_save_debug_log: bool
    is_save_rate_log: bool

    is_available_re_training: bool
    re_training: bool
    load_model_path: str


@dataclass
class GlobalConfig:
    data: dict

    path: Path_ = field(init=False)
    dataset: Dataset_ = field(init=False)
    gradcam: Gradcam_ = field(init=False)
    network: Network_ = field(init=False)
    option: Option_ = field(init=False)

    def __post_init__(self):
        self.path = Path_(**self.data.pop("path"))
        self.dataset = Dataset_(**self.data.pop("dataset"))
        self.gradcam = Gradcam_(**self.data.pop("gradcam"))
        self.network = Network_(**self.data.pop("network"))
        self.option = Option_(**self.data.pop("option"))

        self.filename_base = datetime.now().strftime("%Y%b%d_%Hh%Mm%Ss")


def factory_config(path: str) -> GlobalConfig:
    path = path if path is not None else "us.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
        return GlobalConfig(data)


if __name__ == "__main__":
    with open("us.yaml") as f:
        data = yaml.safe_load(f)
        print(data)

        s = GlobalConfig(data)
        print(s.dataset.limit_size)
