from typing import Dict, Optional


class Model:
    def _old_init__(
            self,
            toml_settings: Optional[_tms.TomlSettings] = None,
            classes: Optional[Dict[int, str]] = None,
            use_gpu: bool = True,
            log: Optional[ul.LogFile] = None,
            rate: Optional[ul.LogFile] = None,
            load_pth_path: Optional[str] = None) -> None:
        """
        Args:
            toml_settings (_tms.TomlSettings): parameters of user's setting.
            classes (Optional[dict], optional): classes of dataset. Defaults to None.
            use_gpu (bool, optional): uisng gpu or cpu. Defaults to False.
            log (Optional[ul.LogFile], optional): logging for debug log. Defaults to None.
            rate (Optional[ul.LogFile], optional): logging for rate log. Defaults to None.
            load_pth_path (Optional[str], optional): pth(model) path. Defaults to None.

        `load_pth_path` is not None -> load pth automatically.
        `classes` is None -> load from 'config/classes.txt' if pth load is False,
                             load from pth checkpoint if pth laod is True.
        `use_gpu` is assigned False automatically if cuda is not available.
        """

        # network configure
        self.net: cnn.Net  # network
        self.optimizer: Union[optim.Adam, optim.SGD]  # optimizer
        self.criterion: nn.CrossEntropyLoss  # criterion

        self.classes: Dict[int, str]  # class
        self.input_size = toml_settings.input_size  # image size when input to network
        self.load_pth_path = load_pth_path  # pth path
        # self.current_epoch: int  # for load pth

        # gpu setting
        self.use_gpu = torch.cuda.is_available() and use_gpu
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        if toml_settings is None:
            toml_settings = _tms.factory()
        self.tms = toml_settings

        # for tensorboard
        self.writer = tbx.SummaryWriter()

        # assign or load from classes.txt
        if load_pth_path is None:
            cls_txt = f'{toml_settings.config_path}/classes.txt'
            self.classes = classes if classes is not None \
                else ul.load_classes(cls_txt).copy()

        # create instance if log is None
        self.log = log if log is not None else ul.LogFile(None)

        # create instance if rate is None
        self.rate = rate if rate is not None else ul.LogFile(None)

        # build model by cnn.py or load model
        self.__build_model()

        # save classes
        self.__write_classes()

        # classes.txt is EOF or classes is None
        if self.classes is {}:
            print('classes is {}')
    # end of [function] __init__

    def __build_model(self):
        r""" Building model.

        * load pth
            -> network: load from pth model.
            -> optimizer: Adam algorithm.
            -> criterion: Cross Entropy Loss algorithm.

            -> classes: load from pth model.
            -> epoch: load from pth model.

        * do not load pth
            -> network: Net(cnn.py) instance and init gradient.
            -> optimizer: Adam algorithm.
            -> criterion: Cross Entropy Loss algorithm.
        """

        self.net = cnn.Net(input_size=self.input_size)  # network

        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=pow(10, -8))
        self.criterion = nn.CrossEntropyLoss()

        # load
        if self.load_pth_path is None:
            self.net.zero_grad()  # init all gradient
        else:
            # check exist
            if not Path(self.load_pth_path).exists():
                err = OSError(errno.ENOENT, os.strerror(errno.ENOENT), self.load_pth_path)
                raise FileNotFoundError(err)

            # load checkpoint
            checkpoint = torch.load(self.load_pth_path)

            # classes, network
            self.classes = checkpoint['classes']
            self.net.load_state_dict(checkpoint['model_state_dict'])

            # self.current_epoch = checkpoint['epoch']
            # criterion = checkpoint['criterion']
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.net.to(self.device)  # switch to GPU / CPU
    # end of [function] __build_model
