from pathlib import Path
import numpy as np
from PIL import Image
import random


class Data:
    def __init__(self, path=None, label=0, name='None'):
        if path is None:
            self.path = None
            return

        self.path = path
        self.label = label
        self.name = name
        self.img_data = np.array(0)

    def load_image(self, loader, device):
        img = Image.open(self.path)
        img = loader(img).unsqueeze(0)
        return img.to(device)

    def configs(self):
        return self.name, self.label, self.path


class Datasets:
    def __init__(
            self,
            path,
            extensions,
            all_size,
            test_size,
            minibatch_size):

        # for instance temporary in Main class
        xx = [path, extensions, all_size, test_size]
        if None in xx:
            return

        self.path = path
        self.extensions = extensions

        # images
        self.train_list = [Data()]
        self.test_list = [Data()]

        self.each_cls_list = {}  # all images
        self.classes = []  # classes of datasets

        # remove first element / for using auto completion
        self.remove_first_element(self.train_list)
        self.remove_first_element(self.test_list)

        # size of images
        self.all_size = all_size
        self.train_size = all_size - test_size
        self.test_size = test_size

        self.minibatch_size = minibatch_size
        self.now_train_size = 0

        # ----------

        # train_list / test_list
        self.get_all_datas()

        # shuffle data
        # self.shuffle_data(target='train')

    def get_all_datas(self):
        """ Get All Datasets from each directory. """
        path = Path(self.path)

        # directories in [image_path]
        dirs = [d for d in path.glob('*') if d.is_dir()]

        # classes
        self.classes = [str(d.name) for d in dirs]

        # all extensions / all sub directories
        threshold = self.test_size

        for idx, _dir in enumerate(dirs):
            xs = []
            for ext in self.extensions:
                tmp = [Data(x.as_posix(), idx, _dir.name)
                       for x in _dir.glob(f'*.{ext}') if x.is_file()]
                xs.extend(tmp)

            train = xs[threshold:]
            test = xs[:threshold]
            self.each_cls_list[_dir.name] = {
                'train': train,
                'test': test
            }

            random.shuffle(train)
            self.train_list.extend(train)
            self.test_list.extend(test)

    def shuffle_data(self, target='train'):
        new_list = [Data(None)]
        self.remove_first_element(new_list)

        for _cls in self.classes:
            xs = self.each_cls_list[_cls][target]
            random.shuffle(xs)
            new_list.extend(xs)

        # shuffle new_list of all
        # random.shuffle(new_list)

        if target == 'train':
            self.train_list = new_list
        elif target == 'test':
            self.test_list = new_list

    def get_next_train_datas(self):
        plus = min(self.minibatch_size, self.train_size - self.now_train_size)

        # if train data is empty, return None
        #  => end of this epoch
        if plus <= 0:
            return [Data(None)]
            # yield [Data(None)]

        bgn = self.now_train_size
        end = bgn + plus

        self.now_train_size += plus
        return self.train_list[bgn:end]
        # yield self.train_list[bgn:end]

    def remove_first_element(self, _list):
        _list.remove(_list[0])

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
