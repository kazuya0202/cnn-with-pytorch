from pathlib import Path


class LogFile:
    def __init__(self, path):
        self.path = Path(path)

        # create output dir
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # create output file
        self.path.touch()

    def write(self, line, debug_ok=True):
        with self.path.open('a') as f:
            ct = line + '\n'
            f.write(ct)

        if debug_ok:
            print(line)
