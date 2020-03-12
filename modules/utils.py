import errno
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


# type annotation
_path_t = Union[str, Path]


@dataclass
class LogFile:
    path: Optional[_path_t] = None
    std_debug_ok: bool = True
    _clear: bool = False

    def __post_init__(self):
        self._std_debug_ok = self.std_debug_ok

        if self.path is None:
            self._is_write = False
            return

        self._path = Path(self.path)
        self._is_write = True

        if self._clear:
            self.clear()

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)

        # open file as append mode.
        self._file = self._path.open("a")

    def write(self, line: object = "", debug_ok: bool = None) -> None:
        if self._is_write:
            self._file.write(str(line))

        if self._debug_ok(debug_ok):
            print(f"\r{line}", end="")

    def writeline(self, line: object = "", debug_ok: bool = None) -> None:
        if self._is_write:
            self._file.write(f"{line}\n")

        if self._debug_ok(debug_ok):
            print(line)

    def clear(self) -> None:
        if not self._is_write:
            return

        if self._path.exists():
            self._path.unlink()  # delete
        self._path.touch()  # create

    def _debug_ok(self, debug_ok: Optional[bool]) -> bool:
        if debug_ok is not None:
            return debug_ok  # priority `debug_ok`
        return self._std_debug_ok

    def close(self) -> None:
        if self._is_write:
            self._file.close()


def create_file_path(
    dir_path: _path_t, name: str, head: str = None, end: str = "", ext="txt"
) -> str:
    parent = Path(dir_path)

    _head = head if head is not None else str(len([x for x in parent.glob(f"*.{ext}")])) + "_"

    path = parent.joinpath(f"{_head}{name}{end}.{ext}")
    return path.as_posix()


@dataclass
class ProgressLog:
    line: str

    def __post_init__(self):
        self.progress()

    def progress(self) -> None:
        print(f"LOG: [running] {self.line} ...", end="", flush=True)

    def complete(self) -> None:
        print(f"\rLOG: [completed] {self.line}. \n")


def make_directories(*dir_list) -> None:
    for dir_path in dir_list:
        if dir_path is None:
            continue

        if not isinstance(dir_path, Path):
            dir_path = Path(dir_path)

        dir_path.mkdir(parents=True, exist_ok=True)


def load_classes(path="config/classes.txt") -> dict:
    path = Path(path)
    classes = {}

    if not path.exists():
        return classes

    lines = path.read_text().split("\n")

    for line in lines:
        _list = line.split(":")
        if len(_list) < 2:
            continue

        x, y = list(map(lambda x: x.strip(), _list))
        classes[int(x)] = y
    return classes


def raise_when_FileNotFound(path: _path_t) -> None:
    if os.path.exists(str(path)):
        return

    err = OSError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    raise FileNotFoundError(err)


# def set_align_ljust(content: str, align: int = 10) -> str:
#     return content.ljust(align)
# def set_align_rjust(content: str, align: int = 10) -> str:
#     return content.rjust(align)
# def set_align_center(content: str, align: int = 10) -> str:
#     return content.center(align)
