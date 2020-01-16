from pathlib import Path


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
    def __init__(self, path):
        self.path = Path(path)

        # create output dir
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True)

        # create output file
        self.path.touch()

    def write(self, line, debug_ok=True):
        with self.path.open('a') as f:
            f.write(line)

        if debug_ok:
            print(f'\r{line}', end='')

    def writeline(self, line, debug_ok=True):
        with self.path.open('a') as f:
            f.write(f'{line}\n')

        if debug_ok:
            print(line)


def create_file_path(dir_path, name, ext='txt'):
    parent = Path(dir_path)
    num = len([x for x in parent.glob(f'*.{ext}')])

    path = parent.joinpath(f'{num}_{name}.{ext}')
    return path
