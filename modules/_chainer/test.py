from chainer import cuda, Variable
import numpy as np
import GlobalVariable as gv
import pickle
from PIL import Image
import chainer.functions as F

from pathlib import Path
import sys


class Test():
    def __init__(self, img_path, pkl_path):
        self.img_path = Path(img_path)
        self.pkl_path = Path(pkl_path)

        # GPU
        gv.G.xp = cuda.cupy if gv.G.IsGPU else np

        self.model = None
        self.env = gv.G.xp
        self.witdh = gv.G.Width
        self.height = gv.G.Height
        self.exts = gv.G.Extention

    def recognize(self, img_path):
        img = Image.open(img_path)
        img = img.resize((self.witdh, self.height))
        img = np.array(img, dtype=self.env.float32)
        img = img[:, :, :3]
        img = img.transpose(2, 0, 1)

        xt = Variable(self.env.array([img], dtype=self.env.float32))
        yy = self.model(xt)
        y = yy['fc8']
        y = F.softmax(y.data)

        ans = int(self.env.argmax(y.data))
        return ans

    def main(self):
        # pickle
        with open(self.pkl_path, 'rb') as p:
            self.model = pickle.load(p)

        print(f'Load pickle: {self.pkl_path}\n')

        # only one image
        if self.img_path.is_file():
            ans = self.recognize(self.img_path)
            print(f'ans: {ans}')
            return

        # directory (multi images)
        # if self.img_path.is_dir():

        print(' --- Configs ---')

        x = {}
        num = 0
        for _dir in self.img_path.glob('*'):
            _dir = Path(_dir)

            if _dir.is_file():
                continue

            name = _dir.name
            x[name] = {
                'img': [],
                'lbl': []
            }

            for ext in self.exts:
                for file in _dir.glob(f'*.{ext}'):
                    x[name]['img'].append(file)
                    x[name]['lbl'].append(num)

            size = len(x[name]['img'])
            print(f'label: {num} / [{name}] ({size} images.)')
            num += 1

        if num == 0:
            print('There is not exist directory.')
            return

        print('\n --- Results ---')

        for key, value in x.items():
            # acc count
            acc_num = 0

            for i, v in enumerate(value['img']):
                # recognize
                ans = self.recognize(v)

                # 正解なら +1
                if ans == x[key]['lbl'][i]:
                    acc_num += 1

                # progress
                print(f'\r # [{key}]: {i}', end='')
            print('\r', end='')

            # accuracy rate
            all_size = len(value['img'])
            if all_size == 0:
                print(f'[{key} acc: ? %] -- ({acc_num} / {all_size} images.)')
                continue

            acc = acc_num / all_size * 100
            print(f'[{key}] acc: {acc} %  --  ({acc_num} / {all_size} images.)')


if __name__ == '__main__':
    # ======= 変更箇所 =========

    # --- Images / pickle ---
    img_path = r'./rec/Images-spectrum'
    pkl_path = r'./rec/0_ymd20191226_hms124959.pickle'

    # ==========================

    argv = sys.argv
    if len(argv) >= 2:
        img_path = argv[1]
    if len(argv) >= 3:
        pkl_path = argv[2]

    img_path = Path(img_path)
    pkl_path = Path(pkl_path)

    if not img_path.exists():
        print(f'There is not exist \'{img_path}\'')
        exit(-1)

    if not pkl_path.exists():
        print(f'There is not exist \'{pkl_path}\'')
        exit(-1)

    test = Test(img_path, pkl_path)
    test.main()
