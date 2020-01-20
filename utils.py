from pathlib import Path
from typing import Optional, Union


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
    # end of [function] __init__

    def write(self, line='', debug_ok: bool = False):
        if self.__is_write():
            with self.path.open('a') as f:
                f.write(line)

        if self.all_debug_ok or debug_ok:
            print(f'\r{line}', end='')
    # end of [function] write

    def writeline(self, line='', debug_ok: bool = False):
        if self.__is_write():
            with self.path.open('a') as f:
                f.write(f'{line}\n')

        if self.all_debug_ok or debug_ok:
            print(line)
    # end of [function] writeline

    def clear_all(self):
        self.path.unlink()  # delete
        self.path.touch()  # create
    # end of [function] clear_all

    def __is_write(self):
        return self.path is not None
    # end of [function] __is_write
# end of [class] LogFile


class DebugRateLogs():
    def __init__(
            self,
            log: Optional[LogFile] = None,
            rate: Optional[LogFile] = None):

        self.log = log if log is None else LogFile(None)
        self.rate = rate if rate is None else LogFile(None)
    # end of [function] __init__

    def set_log(self, log: LogFile):
        self.log = log
    # end of [function] set_log

    def set_rate(self, rate: LogFile):
        self.rate = rate
    # end of [function] set_rate
# end of [class] DebugRateLogs


def create_file_path(
        dir_path: Union[str, Path],
        name: str,
        head: Optional[str] = None,
        ext='txt'):

    parent = Path(dir_path)

    _head = head if head is not None \
        else str(len([x for x in parent.glob(f'*.{ext}')])) + '_'

    path = parent.joinpath(f'{_head}{name}.{ext}')
    return path.as_posix()
# end of [function] create_file_path


class DebugLog:
    def __init__(self, line):
        self.line = line
        self.progress()
    # end of [function] __init__

    def progress(self, bottom=''):
        print(f'LOG: [running] {self.line} ... {bottom}', end='', flush=True)
    # end of [function] progress

    def complete(self):
        print(f'\rLOG: [completed] {self.line}. \n')
    # end of [function] complete


def make_directories(*args):
    for p in args:
        if p is None:
            continue

        p = Path(p)
        if not p.exists():
            p.mkdir(parents=True)
# end of [function] make_directories


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
# end of [function] load_classes


def write_classes(classes: dict, path='./classes.txt'):
    cls_file = LogFile(path, default_debug_ok=False)


    for k, _cls in classes.items():
        cls_file.writeline(f'{k}:{_cls}')
# end of [function] write_classes


def write_used_image_path(_list: list, path='./used_images.txt'):
    _file = LogFile(path, default_debug_ok=False)

    for x in _list:
        _file.writeline(x.path)
