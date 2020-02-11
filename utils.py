import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorboardX as tbx
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable


@dataclass(init=False)
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

        # clear
        if clear:
            self.clear_all()

        # create output dir
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # create output file
        if not self.path.exists():
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


@dataclass(init=False)
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
        end: str = '',
        ext='txt'):

    parent = Path(dir_path)

    _head = head if head is not None \
        else str(len([x for x in parent.glob(f'*.{ext}')])) + '_'

    path = parent.joinpath(f'{_head}{name}{end}.{ext}')
    return path.as_posix()
# end of [function] create_file_path


@dataclass(init=False)
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
        p.mkdir(parents=True, exist_ok=True)
# end of [function] make_directories


def load_classes(path='config/classes.txt'):
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


@dataclass(init=False)
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


# ----- pytorch -----


def add_confusion_matrix_to_tensorboard(
        writer: tbx.SummaryWriter,
        correct_labels: Union[torch.Tensor, np.ndarray],
        predicted_labels: Union[torch.Tensor, np.ndarray],
        classes: List[str],
        current_epoch: int):

    cm = calc_confusion_matrix(correct_labels, predicted_labels, len(classes))
    fig = plot_confusion_matrix(cm, classes)
    add_to_tensorboard(writer, fig, current_epoch)


def add_to_tensorboard(
        writer: tbx.SummaryWriter,
        fig: plt.figure,
        step: int):

    fig.canvas.draw()
    img = fig.canvas.renderer._renderer
    img_ar = np.array(img).transpose(2, 0, 1)

    writer.add_image('confusion matrix', img_ar)
    plt.close()


def plot_confusion_matrix(
        cm: Union[torch.Tensor, np.ndarray],
        classes: List[str],
        normalize: bool = False,
        title: str = 'Confusion matrix',
        cmap: plt.cm = plt.cm.Greens):

    # torch.Tensor to np.ndarray
    _cm: np.ndarray = cm if not isinstance(cm, torch.Tensor) \
        else cm.cpu().numpy()

    if normalize:
        _cm = _cm.astype('float') / _cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # change font size
    plt.rcParams["font.size"] = 18

    fig, axes = plt.subplots(figsize=(10, 10))

    # ticklabels
    tick_marks = np.arange(len(classes))

    plt.setp(axes, xticks=tick_marks, xticklabels=classes)
    plt.setp(axes, yticks=tick_marks, yticklabels=classes)
    # rotate xticklabels
    plt.setp(axes.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # title
    plt.suptitle('Confusion Matrix')

    # label
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')

    # grid
    # axes.grid(which='minor', color='b', linestyle='-', linewidth=3)

    img = plt.imshow(cm, interpolation='nearest', cmap=cmap)

    # adjust color bar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(img, cax=cax)

    thresh = cm.max() / 2.
    fmt = '.2f' if normalize else 'd'

    # plot text
    for i, j in itertools.product(range(len(classes)), range(len(classes))):
        clr = 'white' if cm[i, j] > thresh else 'black'
        axes.text(j, i, format(cm[i, j], fmt), ha='center', va='center', color=clr)

    # plt.show()
    plt.tight_layout()
    fig = plt.gcf()
    return fig


def calc_confusion_matrix(
        correct_labels: Union[torch.Tensor, np.ndarray],
        predicted_labels: Union[torch.Tensor, np.ndarray],
        class_num: int):

    cm = torch.zeros(class_num, class_num, dtype=torch.int64)
    stacked = torch.stack((correct_labels, predicted_labels), dim=1)

    for p in stacked:
        tl, pl = p.tolist()
        cm[tl, pl] += 1

    return cm
