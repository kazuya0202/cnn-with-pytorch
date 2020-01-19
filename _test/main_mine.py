from pathlib import Path
import datetime
import numpy as np

# torch
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import torchvision.transforms as transforms

# my packages
import utils as ul
from global_variables import GlobalVariables
from datasets import Datasets
import cnn


class Main():
    def __init__(self):
        self.gv = GlobalVariables()

        # datasets
        self.ds = Datasets(*([None] * 5))

    def execute(self):
        net = cnn.Net()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # use_cudnn = torch.backends.cudnn.version()

        # print(device)
        # print(use_cudnn)

        net.to(device)
        print(net)

        # _input = torch.randn(1, 1, 32, 32).to(device)
        # out = net(_input)
        # print(out)

        net.zero_grad()
        # out.backward(torch.randn(1, 10).to(device))

        image_size = 80
        loader = transforms.Compose([
            # transforms.Resize(32),
            transforms.ToTensor()])

        def load_image(path, loader, device):
            from PIL import Image
            img = Image.open(path)
            img = img.convert('RGB')
            img = img.resize((32, 32))

            # img = loader(img).unsqueeze(0)
            img = loader(img)
            return img.to(device)

        p = r'C:\ichiya\repos\github.com\kazuya0202\cnn-with-pytorch\recognition_datasets\Images\crossing\crossing-samp1_10_1.jpg'
        s = load_image(p, loader, device)
        print(s.size())
        # exit()
        # t = torch.FloatTensor([s])
        # print(t)
        # out = net()
        # print(out)
        exit()

        # if not exist, exit script
        if not Path(self.gv.image_path).exists():
            print(f'The directory \'{self.gv.image_path}\' does not exist.')
            exit()

        # ===== create datasets =====
        log = ul.DebugLog('Getting datasets')

        # arguments of class(Datasets)
        params = [
            self.gv.image_path,  # path
            self.gv.extensions,  # extensions
            self.gv.dataset_size,  # all_size
            self.gv.test_size,  # test_size
            self.gv.batch_size,  # minibatch_size
        ]

        self.ds = Datasets(*params)
        log.complete()

        # ===== create make required direcotry =====
        log = ul.DebugLog('Making required directory')

        # argments
        params = [
            self.gv.log_path,
        ]

        ul.make_directories(*params)
        log.complete()

        print()
        self.ds.print_parameter_config()

        # get file name
        self.dt_now = str(datetime.datetime.now().strftime(
            "ymd%Y%m%d_hms%H%M%S")).replace(" ", "_")
        print(self.dt_now)

        # usage: log file
        # params = [self.gv.log_path, dt_now]
        # t = ul.create_file_path(*params)
        # s1 = ul.LogFile(t)
        # t = ul.create_file_path(*params, 'csv')
        # s2 = ul.LogFile(t)

        # self.usage_test()

    def usage_test(self):
        # --- TRAIN ---
        train_size = self.gv.dataset_size - self.gv.test_size
        # while True:
        _ceil = int(np.ceil(train_size / self.gv.batch_size))
        for i in range(_ceil):
            print(f'\nloop times:', i + 1)

            for x in self.ds.get_next_train_datas():
                # end of this epoch
                if x.path is None:
                    return
                print(x.path)
        print('\nend this epoch.')

        # xs = self.ds.get_next_train_datas()
        # for x in xs:
        #     print(x.path)

        # params = [self.gv.log_path, self.dt_now]
        # t = ul.create_file_path(*params)
        # s1 = ul.LogFile(t)

        # usage: configs
        # print('train list:')
        # for d in self.ds.train_list:
        #     # s1.writeline(d.configs())
        #     print(d.configs())

        # print('test list:')
        # for d in ds.test_list:
        #     print(d.configs())
        # -----

        # usage: list_each_class
        # for c in ds.classes:
        #     for k, v in ds.each_cls_list[c].items():
        #         print(f'key: {k}')
        #         for x in v:
        #             print(x.path)
        # -----


# def deco(func):
#     def wrapper(*args, **kwargs):
#         print(*args)
#         print('a')
#         func(*args, **kwargs)
#         print('c')
#     return wrapper


# @deco
# def testa(a):
#     print('b')


if __name__ == '__main__':
    # testa('b')
    # exit()

    main = Main()
    main.execute()
