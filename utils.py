from pathlib import Path
from typing import Optional, Union


class LogFile():
    def __init__(
            self,
            path: Optional[str],
            default_debug_ok: bool = True,
            clear: bool = False):

        self.all_debug_ok = default_debug_ok
        if path is None:
            self.path = path
            return

        self.path = Path(path)

        if clear:
            # clear
            self.clear_all()

        # create output dir
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True)

        if not self.path.exists():
            # create output file
            self.path.touch()
    # end of [function] __init__

    def write(self, line='', debug_ok: Optional[bool] = None):
        if self.__is_write():
            with self.path.open('a') as f:
                f.write(line)

        if self.__debug_ok(debug_ok):
            print(f'\r{line}', end='')
    # end of [function] write

    def writeline(self, line='', debug_ok: Optional[bool] = None):
        if self.__is_write():
            with self.path.open('a') as f:
                f.write(f'{line}\n')

        if self.__debug_ok(debug_ok):
            print(line)
    # end of [function] writeline

    def clear_all(self):
        self.path.unlink()  # delete
        self.path.touch()  # create
    # end of [function] clear_all

    def __is_write(self):
        return self.path is not None
    # end of [function] __is_write

    def __debug_ok(self, debug_ok):
        if debug_ok is not None:
            # priority `debug_ok`
            if debug_ok:
                return True
        elif self.all_debug_ok:
            return True

        return False
    # end of [function] __debug_ok
# end of [class] LogFile


class DebugRateLogs():
    def __init__(
            self,
            log: Optional[LogFile] = None,
            rate: Optional[LogFile] = None):

        self.log = log if log is not None else LogFile(None, True)
        self.rate = rate if rate is not None else LogFile(None, True)
    # end of [function] __init__

    def set_log(self, log: LogFile):
        self.log = log
    # end of [function] set_log

    def set_rate(self, rate: LogFile):
        self.rate = rate
    # end of [function] set_rate
# end of [class] DebugRateLogs


def new(name: str, data: dict):
    """
    Usage:
        obj = new('Obj', {'x': 1, 'y': 2})

        print(obj.x)
        print(obj.y)
    """
    return type(name, (object,), data)
# end of [function] new


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


class ProgressLog():
    def __init__(self, line):
        self.line = line
        self.progress()
    # end of [function] __init__

    def progress(self):
        print(f'LOG: [running] {self.line} ...', end='', flush=True)
    # end of [function] progress

    def complete(self):
        print(f'\rLOG: [completed] {self.line}. \n')
    # end of [function] complete
# end of [class] ProgressLog


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


class RunningObject():
    """
    Usage:
        robj = RunningObject('testing ...')
        for ... :
            robj.flush()
        robj.finish()
    """

    def __init__(self, head: str = ''):
        self.cnt = 0
        self.head = head
        self.content = ['\\', '-', '/', '-']

    def finish(self):
        print()

    def __next_obj(self):
        idx = self.cnt % 4
        obj = self.content[idx]
        self.cnt += 1

        return obj

    def flush(self):
        obj = self.__next_obj()
        print(f'\r{self.head} {obj}', end='', flush=True)
