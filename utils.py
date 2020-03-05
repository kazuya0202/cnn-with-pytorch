import errno
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import numpy as np


@dataclass(init=False)
class LogFile:
    def __init__(self, path: Optional[Union[str, Path]],
                 std_debug_ok: bool = True, clear: bool = False) -> None:

        self.std_debug_ok = std_debug_ok
        self._is_write = True

        if path is None:
            self.path = path
            self._is_write = False
            return

        self.path = Path(path)

        if clear:
            self.clear()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

        # open file as append mode.
        self._file = self.path.open('a')
    # end of [function] __init__

    def write(self, line: object = '', debug_ok: Optional[bool] = None) -> None:
        if self._is_write:
            self._file.write(str(line))

        if self._debug_ok(debug_ok):
            print(f'\r{line}', end='')
    # end of [function] write

    def writeline(self, line='', debug_ok: Optional[bool] = None) -> None:
        if self._is_write:
            self._file.write(f'{line}\n')

        if self._debug_ok(debug_ok):
            print(line)
    # end of [function] writeline

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()  # delete
        self.path.touch()  # create
    # end of [function] clear

    def _debug_ok(self, debug_ok: Optional[bool]) -> bool:
        if debug_ok is not None:
            return debug_ok  # priority `debug_ok`
        return self.std_debug_ok
    # end of [function] _debug_ok

    def close(self) -> None:
        self._file.close()
    # end of [function] close
# end of [class] LogFile


def create_file_path(dir_path: Union[str, Path], name: str, head: Optional[str] = None,
                     end: str = '', ext='txt') -> str:
    parent = Path(dir_path)

    _head = head if head is not None \
        else str(len([x for x in parent.glob(f'*.{ext}')])) + '_'

    path = parent.joinpath(f'{_head}{name}{end}.{ext}')
    return path.as_posix()
# end of [function] create_file_path


@dataclass(init=False)
class ProgressLog():
    def __init__(self, line: str) -> None:
        self.line = line
        self.progress()
    # end of [function] __init__

    def progress(self) -> None:
        print(f'LOG: [running] {self.line} ...', end='', flush=True)
    # end of [function] progress

    def complete(self) -> None:
        print(f'\rLOG: [completed] {self.line}. \n')
    # end of [function] complete
# end of [class] ProgressLog


def make_directories(*dir_list) -> None:
    for dir_path in dir_list:
        if dir_path is None:
            continue

        if not isinstance(dir_path, Path):
            dir_path = Path(dir_path)

        dir_path.mkdir(parents=True, exist_ok=True)
# end of [function] make_directories


def load_classes(path='config/classes.txt') -> dict:
    path = Path(path)
    classes = {}

    if not path.exists():
        # raise FileNotFoundError
        return classes

    lines = path.read_text().split('\n')

    for line in lines:
        _list = line.split(':')
        if len(_list) < 2:
            continue

        x, y = list(map(lambda x: x.strip(), _list))
        classes[int(x)] = y

    return classes
# end of [function] load_classes


def find_str(_str: str, keyword: str) -> bool:
    if _str.find(keyword) > -1:
        return True

    return False
# end of [function] find_str


def raise_when_FileNotFound(path) -> None:
    if os.path.exists(str(path)):
        return

    err = OSError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    raise FileNotFoundError(err)
# end of [function] raise_when_FileNotFound


# def set_align_ljust(content: str, align: int = 10) -> str:
#     return content.ljust(align)
# def set_align_rjust(content: str, align: int = 10) -> str:
#     return content.rjust(align)
# def set_align_center(content: str, align: int = 10) -> str:
#     return content.center(align)
