from pathlib import Path
from typing import Optional, Union


class TorchModel:
    import torch
    # def __init__(self):
    #     pass

    def save(self, path, model):
        # save info
        # => epoch ...
        pass

    def load(self, path):
        # self.epoch = None
        pass


class LogFile:
    def __init__(self, path: Optional[str], default_debug_ok: bool = True):
        if path is None:
            self.path = path
            return

        self.path = Path(path)
        self.all_debug_ok = default_debug_ok

        # create output dir
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True)

        # create output file
        self.path.touch()

    def write(self, line='', debug_ok: bool = False):
        if self.__is_write():
            with self.path.open('a') as f:
                f.write(line)

        if self.all_debug_ok or debug_ok:
            print(f'\r{line}', end='')

    def writeline(self, line='', debug_ok: bool = False):
        if self.__is_write():
            with self.path.open('a') as f:
                f.write(f'{line}\n')

        if self.all_debug_ok or debug_ok:
            print(line)

    def __is_write(self):
        return self.path is not None


def create_file_path(
        dir_path: Union[str, Path],
        name: str,
        head: Optional[str] = None,
        ext='txt'):

    parent = Path(dir_path)

    _head = head if head is not None \
        else len([x for x in parent.glob(f'*.{ext}')])

    path = parent.joinpath(f'{_head}_{name}.{ext}')
    return path.as_posix()


class DebugLog:
    def __init__(self, line):
        self.line = line
        self.progress()

    def progress(self, bottom=''):
        print(f'LOG: [running] {self.line} ... {bottom}', end='', flush=True)

    def complete(self):
        print(f'\rLOG: [completed] {self.line}. \n')


def make_directories(*args):
    for p in args:
        if p is None:
            continue

        p = Path(p)
        if not p.exists():
            p.mkdir(parents=True)


def load_classes(path='./classes.txt'):
    path = Path(path)
    classes = {}

    text = path.read_text()
    lines = text.split('\n')

    for line in lines:
        x, y = line.split(':')
        x = x.strip()
        y = y.strip()

        classes[x] = y

    return classes


def write_classes(classes: dict, path='./classes.txt'):
    cls_file = LogFile(path, default_debug_ok=False)

    with open(path, 'w'):
        for k, _cls in classes.items():
            cls_file.writeline(f'{k}:{_cls}')
