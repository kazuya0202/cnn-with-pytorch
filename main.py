from pathlib import Path
import datetime

# my packages
import utils as ul
from global_variables import GlobalVariables
from datasets import Datasets


class Main():
    def __init__(self):
        self.gv = GlobalVariables()

        # datasets
        # self.ds = Datasets(*([None] * 5))

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
        ds = Datasets(*params)

        # get file name
        dt_now = str(datetime.datetime.now().strftime(
            "ymd%Y%m%d_hms%H%M%S")).replace(" ", "_")
        print(dt_now)

        xs = ds.get_next_train_datas()
        for x in xs:
            print(x.path)

        # usage: configs
        # for d in ds.train_list:
        #     print(d.configs())

        # usage: list_each_class
        # for c in ds.classes:
        #     for k, v in ds.list_each_class[c].items():
        #         print(k, v)


if __name__ == '__main__':
    main = Main()
    main.execute()
