class GlobalVariables:
    def __init__(self):
        # ===== USER SETTINGS =====

        """ データセット """
        # 各データセットの枚数
        # self.dataset_size = 40

        # 各データセットのテスト比率(枚数) ↓どちらか一方を使う
        self.test_size: float = 0.1  # 比率: (0.0 ~ 1.0) type[float]
        # self.test_size: int = 400  # 枚数: (1 ~ N)  type[int]

        # 入力画像サイズ ↓どちらか一方を使う
        # self.image_size: tuple = (60, 60)  # 縦横が違う場合
        self.image_size: int = 60  # 縦横が同じ場合は省略可

        self.epoch = 2  # エポック
        self.batch_size = 100  # バッチ(何枚ずつ行うか)
        self.extensions = ['jpg', 'png', 'jpeg', 'bmp', 'gif']  # 拡張子

        self.is_shuffle_per_epoch = True  # エポック毎にデータセットをシャッフルするかどうか

        # [cycle]: 0 -> 何もしない / 10 -> 10 epoch / N -> N epoch ...
        self.pth_save_cycle = 2  # 学習モデル(pth) の保存サイクル('{self.pth_path}/epoch_pth/' に保存)
        self.test_cycle = 1  # 学習モデルの test サイクル

        # 使用する画像(各クラスごとにフォルダにまとめて、そのフォルダをまとめたパスを指定)
        # self.image_path = r'./recognition_datasets/Images/'
        self.image_path = r'C:\ichiya\prcn2019\prcn2019-datasets\datasets\Images-20191014'
        # self.image_path = r'D:\workspace\repos\gitlab.com\ichiya\prcn2019-datasets\datasets\Images-20191014'

        # 間違えた画像の保存先
        self.false_path = r'./recognition_datasets/False/'

        # 学習モデルの保存先
        self.pth_path = r'./recognition_datasets/'

        """ ログ """
        self.is_save_debug_log = True  # コンソールの出力をファイルに書き出すかどうか
        self.is_save_rate_log = True  # エポック毎の認識率を保存するかどうか

        # ログ(学習状況)の出力先
        self.log_path = r'./recognition_datasets/Logs/'

        """ Grad CAM """
        self.is_grad_cam = False  # GradCAM を行うかどうか

        # Grad CAM 画像の保存先
        self.grad_cam_path = r''

        """ 作る？ """
        # self.target_layer = 'fc8'
        # self.grad_cam_layer = 'conv5'

        # ===== COMPLEX SETTINGS =====
        from datetime import datetime
        # self.filename_base = str(datetime.datetime.now().strftime(
        #     "ymd%Y%m%d_hms%H%M%S")).replace(" ", "_")
        self.filename_base = datetime.now().strftime('%Y%b%d_%Hh%Mm%Ss')

        self.use_gpu = True

        """ want to do (unimplemented) """
        import torch.optim as optim
        self.optimizer = optim.Adam
        # self.optim = optim.SGD

        # ↓多分これは変わらないからなくてもいい？
        # import torch.nn as nn
        # self.criterion = nn.CrossEntropyLoss()

    # end of [function] __init__
# end of [class] GlobalVariables
