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
        self.ds = Datasets(*([None] * 5))

    def execute(self):
        # path = self.gv.image_path
        # exts = self.gv.extensions
        # all_size = self.gv.image_size
        # train_size = self.gv.train_size
        # test_size = self.gv.test_size

        # not exist
        if not Path(self.gv.image_path).exists():
            print(f'The directory \'{self.gv.image_path}\' does not exist.')

        # arguments of Datasets
        params = [
            self.gv.image_path,  # path
            self.gv.extensions,  # extensions
            self.gv.image_size,  # all_size
            self.gv.train_size,  # train_size
            self.gv.test_size,  # test_size
            True    # is_print_cfg
        ]

        # self.ds = Datasets(path, exts, all_size, train_size, test_size, True)
        self.ds = Datasets(*params)

        for d in self.ds.train_list:
            print(d.configs())

        # datetime
        # print(datetime.datetime.now())


if __name__ == '__main__':
    main = Main()
    main.execute()
