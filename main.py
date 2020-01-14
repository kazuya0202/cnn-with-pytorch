from pathlib import Path

# my packages
import utils as ul
from global_variables import GlobalVariables


class Main():
    def __init__(self):
        self.gv = GlobalVariables()

        # file config
        self.files = []
        self.dirs = []

    def run(self):
        image_path = self.gv.image_path
        exts = self.gv.extensions

        self.files, self.dirs = ul.get_datasets(image_path, exts)

        print('----- Configs -----')
        print(f'{image_path}')
        for i, _dir in enumerate(self.dirs):
            print(f'    |- {_dir.stem} : label [{i}]')
        print()

        print(f'* image size:', len(self.files))
        print('-------------------')

if __name__ == '__main__':
    main = Main()
    main.run()
