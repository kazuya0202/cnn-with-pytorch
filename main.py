from pathlib import Path
import datetime
import numpy as np

# my packages
import utils as ul
from global_variables import GlobalVariables
from datasets import Datasets


class Main():
    def __init__(self):
        self.gv = GlobalVariables()

        # datasets
        self.ds = Datasets(*([None] * 6))

    def execute(self):
        # path = self.gv.image_path
        # exts = self.gv.extensions
        # all_size = self.gv.image_size
        # train_size = self.gv.train_size
        # test_size = self.gv.test_size

        # if not exist, exit script
        if not Path(self.gv.image_path).exists():
            print(f'The directory \'{self.gv.image_path}\' does not exist.')
            exit()

        # arguments of class(Datasets)
        params = [
            self.gv.image_path,  # path
            self.gv.extensions,  # extensions
            self.gv.image_size,  # all_size
            self.gv.test_size,  # test_size
            self.gv.minibatch_size,  # minibatch_size
            True    # is_print_cfg | for debug
            # False    # is_print_cfg
        ]
        self.ds = Datasets(*params)

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

        self.usage_test()

    def usage_test(self):
        # --- TRAIN ---
        train_size = self.gv.image_size - self.gv.test_size
        # while True:
        _ceil = int(np.ceil(train_size / self.gv.minibatch_size))
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


if __name__ == '__main__':
    main = Main()
    main.execute()
