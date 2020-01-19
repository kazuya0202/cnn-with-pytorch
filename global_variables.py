class GlobalVariables:
    def __init__(self):
        # ===== USER SETTINGS =====

        """ データセット """
        # 各データセットの枚数
        # self.dataset_size = 40

        # 各データセットのテスト比率(枚数) ↓どちらか一方を使う
        self.test_size = 0.1  # 比率: (0.0 ~ 1.0) type[float]
        # self.test_size = 400  # 枚数: (1 ~ N)  type[int]

        # 入力画像サイズ ↓どちらか一方を使う
        # self.image_size = (60, 60)  # 縦横が違う場合
        self.image_size = 60  # 縦横が同じ場合は省略可

        self.epoch = 5  # エポック
        self.batch_size = 120  # バッチ(何枚ずつ行うか)
        self.extensions = ['jpg', 'png', 'jpeg', 'bmp', 'gif']  # 拡張子

        self.is_test_per_epoch = True  # エポック毎にテストするかどうか

        # 何エポックごとに pickle を保存するかどうか('{self.pth_path}/epoch_pth' に保存される)
        self.pth_save_cycle = 0  # 0 -> 保存しない / 10 -> 10エポックごと ...

        # 使用する画像(各クラスごとにフォルダにまとめて、そのフォルダをまとめたパスを指定)
        # self.image_path = r'./recognition_datasets/Images/'
        self.image_path = r'D:\workspace\repos\gitlab.com\ichiya\prcn2019-datasets\datasets\Images-20191014'

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
        """ want to do """
        import torch.optim as optim
        self.optimizer = optim.Adam
        # self.optim = optim.SGD

        # ↓多分これは変わらないからなくてもいい？
        # import torch.nn as nn
        # self.criterion = nn.CrossEntropyLoss()
