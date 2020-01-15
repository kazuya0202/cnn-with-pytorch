from pathlib import Path
import numpy as np
from PIL import Image
import random


class Data:
    def __init__(self, path, label=0, name='None'):
        if path is None:
            return

        self.path = Path(path)
        self.label = label
        self.name = name
        self.img_data = np.array(0)

    def open_image(self):
        img = Image.open(self.path)
        # ...
        self.img_data = img

    def configs(self):
        return self.name, self.label, self.path.as_posix()


class Datasets:
    def __init__(
            self,
            path,
            extensions,
            all_size,
            train_size,
            test_size,
            is_print_cfg=False):
        # for instance temporary in Main class
        xx = [path, extensions, all_size, train_size, test_size]
        if None in xx:
            return

        self.path = path
        self.extensions = extensions

        # images
        self.all_list = []
        self.train_list = [Data(None)]
        self.test_list = [Data(None)]

        # size of images
        self.all_size = all_size
        self.train_size = train_size
        self.test_size = test_size

        # classes of datasets
        self.classes = []

        # ----------

        self.get_all_datas()
        self.re_shuffle(is_all=True)
        if is_print_cfg:
            self.print_parameter_config()

    def get_all_datas(self):
        """ Get All Datasets.
        """
        path = Path(self.path)

        # directories in [image_path]
        dirs = [d for d in path.glob('*') if d.is_dir()]

        files = [Data(None)]  # file path list
        files.remove(files[0])  # for using auto completion

        # all extensions / all sub directories
        for idx, _dir in enumerate(dirs):
            for ext in self.extensions:
                target = _dir.glob(f'*.{ext}')
                tmp = [Data(x.as_posix(), idx, _dir.name)
                       for x in target if x.is_file()]
                files.extend(tmp)

        # assign
        self.all_list = files
        self.classes = [str(d.name) for d in dirs]

    def re_shuffle(self, is_all=False):
        """ shuffle list

        Args:
            is_all (bool, optional): target of shuffle. Defaults to False.
        """

        threshold = self.test_size

        if is_all:
            # all shuffle
            sample = random.sample(self.all_list, len(self.all_list))
            self.train_list = sample[threshold:]
            self.test_list = sample[:threshold]
        else:
            # train shuffle
            random.shuffle(self.train_list)

    def get_next_train_datas(self):
        pass

    def print_parameter_config(self):
        print('----- Parameters / Configs -----')
        print(f'* image path:', self.path)
        print(f'* extension:', self.extensions)
        print(f'* image size:', self.all_size)
        print(f'* train size:', self.train_size)
        print(f'* test size:', self.test_size)

        print(f'\n{self.path}')
        for i, x in enumerate(self.classes):
            print(f'    |- {x} : label [{i}]')
        print('--------------------------------\n')
