from pathlib import Path
import numpy as np
from PIL import Image
import random


class Data:
    def __init__(self, path=None, label=0, name='None'):
        if path is None:
            return

        self.path = path
        self.label = label
        self.name = name
        self.img_data = np.array(0)

    def open_image(self):
        img = Image.open(self.path)
        # ...
        self.img_data = img

    def configs(self):
        return self.name, self.label, self.path


class Datasets:
    def __init__(
            self,
            path,
            extensions,
            all_size,
            test_size,
            minibatch_size,
            is_print_cfg=False):

        # for instance temporary in Main class
        xx = [path, extensions, all_size, test_size]
        if None in xx:
            return

        self.path = path
        self.extensions = extensions

        # images
        self.all_list = []
        self.train_list = [Data()]
        self.test_list = [Data()]

        # remove first element / for using auto completion
        self.train_list.remove(self.train_list[0])
        self.test_list.remove(self.test_list[0])

        # size of images
        self.all_size = all_size
        self.train_size = all_size - test_size
        self.test_size = test_size

        self.minibatch_size = minibatch_size

        # classes of datasets
        self.classes = []

        self.each_cls_list = {}
        self.now_train_size = 0
        # self.list_each_class = {}

        # ----------

        # all_list / train_list / test_list
        self.get_all_datas()
        self.classify_datas()
        # self.re_shuffle(is_all=True)

        if is_print_cfg:
            self.print_parameter_config()

    # def get_all_datas(self):
    #     """ Get All Datasets from each directory.
    #     """
    #     path = Path(self.path)

    #     # directories in [image_path]
    #     dirs = [d for d in path.glob('*') if d.is_dir()]

    #     files = [[Data()]]  # file path list
    #     files.remove(files[0])  # for using auto completion

    #     # all extensions / all sub directories
    #     for idx, _dir in enumerate(dirs):
    #         xs = []
    #         for ext in self.extensions:
    #             target = _dir.glob(f'*.{ext}')
    #             tmp = [Data(x.as_posix(), idx, _dir.name)
    #                    for x in target if x.is_file()]
    #             # files.extend(tmp)
    #             xs.extend(tmp)
    #         files.append(xs)

    def get_all_datas(self):
        """ Get All Datasets from each directory.
        """
        path = Path(self.path)

        # directories in [image_path]
        dirs = [d for d in path.glob('*') if d.is_dir()]

        files = [[Data()]]  # file path list
        files.remove(files[0])  # for using auto completion

        # all extensions / all sub directories

        threshold = self.test_size

        for idx, _dir in enumerate(dirs):
            xs = []
            for ext in self.extensions:
                target = _dir.glob(f'*.{ext}')
                tmp = [Data(x.as_posix(), idx, _dir.name)
                       for x in target if x.is_file()]
                xs.extend(tmp)

            _train = xs[threshold:]
            _test = xs[:threshold]
            self.each_cls_list[_dir.name] = {
                'train': _train,
                'test': _test
            }
        self.all_list = files
        self.classes = [str(d.name) for d in dirs]

    def classify_datas(self):
        # def extend_train_test_datas(self):
        for _cls in self.classes:
            xs = self.each_cls_list[_cls]
            random.shuffle(xs['train'])

            self.train_list.extend(xs['train'])
            self.test_list.extend(xs['test'])

        # shuffle in train(or test) list of all
        # random.shuffle(self.train_list)
        # random.shuffle(self.test_list)

    # def classify_datas(self):
    #     threshold = self.test_size

    #     for _cls, x in zip(self.classes, self.all_list):
    #         # shuffle in each class
    #         random.shuffle(x)

    #         # classify
    #         _train = x[threshold:]
    #         _test = x[:threshold]

    #         # save data per each class
    #         self.list_each_class[_cls] = {
    #             'train': _train,
    #             'test': _test
    #         }

    #         self.train_list.extend(_train)
    #         self.test_list.extend(_test)

    #     # shuffle in train(or test) list of all
    #     # random.shuffle(self.train_list)
    #     # random.shuffle(self.test_list)

    # for [all_list] use as one dimension
    # def get_each_test_datas(self):
    #     tmp = []

    #     # all class
    #     for i in range(len(self.classes)):
    #         # get data of each class
    #         _filter = list(filter(lambda x: x.label == i, self.all_list))

    #         # get test data in class data at random
    #         sample = random.sample(_filter, self.test_size)
    #         tmp.extend(sample)
    #     self.test_list = tmp

    def get_next_train_datas(self):
        plus = min(self.minibatch_size, self.train_size - self.now_train_size)
        bgn = self.now_train_size
        end = bgn + plus

        self.now_train_size += plus
        return self.train_list[bgn:end]

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
