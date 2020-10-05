from datetime import datetime
from typing import List, Optional, Tuple, Union
import yaml
from dataclasses import dataclass, field

from . import utils


@dataclass
class Path_:
    dataset: str = r"./dataset"
    mistaken: str = r"./mistaken"
    model: str = r"./"
    config: str = r"./config"
    log: str = r"./logs"
    gradcam: str = r"./GradCAM_results"


@dataclass
class Dataset_:
    limit_size: Optional[int] = -1
    test_size: Union[int, float] = 0.1
    extensions: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.limit_size == -1:
            self.limit_size = None

        if not self.extensions:
            self.extensions = ["jpg", "png", "jpeg"]


@dataclass
class Gradcam_:
    enabled: bool = False
    only_mistaken: bool = True
    layer: str = "conv5"


@dataclass
class Network_:
    height: int = 60
    width: int = 60
    channels: int = 3

    epoch: int = 10
    batch: int = 128
    subdivision: int = 4

    save_cycle: int = 0
    test_cycle: int = 1

    gpu_enabled: bool = True
    is_save_final_model: bool = True
    is_shuffle_dataset_per_epoch: bool = True

    input_size: Tuple[int, int] = field(init=False)  # height, width

    def __post_init__(self) -> None:
        self.input_size = (self.height, self.width)


@dataclass
class Option_:
    is_show_network_difinition: bool = True

    is_debug: bool = False
    is_save_debug_log: bool = True
    is_save_rate_log: bool = True
    is_save_mistaken_pred: bool = False
    is_save_config: bool = True
    is_save_runs: bool = False

    is_available_re_training: bool = False
    re_training: bool = False
    load_model_path: str = r""


@dataclass
class GlobalConfig:
    __data: dict

    path: Path_ = field(init=False)
    dataset: Dataset_ = field(init=False)
    gradcam: Gradcam_ = field(init=False)
    network: Network_ = field(init=False)
    option: Option_ = field(init=False)

    filename_base: str = field(init=False)
    log: utils.LogFile = field(init=False)
    rate: utils.LogFile = field(init=False)

    def __post_init__(self):
        if self.__data == {}:
            return

        path_ = self.__data.pop("path")
        dataset_ = self.__data.pop("dataset")
        gradcam_ = self.__data.pop("gradcam")
        network_ = self.__data.pop("network")
        option_ = self.__data.pop("option")

        self.path = Path_(**path_) if path_ is not None else Path_()
        self.dataset = Dataset_(**dataset_) if dataset_ is not None else Dataset_()
        self.gradcam = Gradcam_(**gradcam_) if gradcam_ is not None else Gradcam_()
        self.network = Network_(**network_) if network_ is not None else Network_()
        self.option = Option_(**option_) if option_ is not None else Option_()

        self.log = utils.LogFile()
        self.rate = utils.LogFile()
        self.filename_base = datetime.now().strftime("%Y%b%d_%Hh%Mm%Ss")


def factory_config(path: str) -> GlobalConfig:
    path = path if path is not None else "user_config.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return GlobalConfig(data)
