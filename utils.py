def get_datasets(image_path, extensions=['jpg']):
    from pathlib import Path

    path = Path(image_path)
    # not exist
    if not path.exists():
        print(f'The directory \'{image_path}\' does not exist.')

    # directories in [image_path]
    dirs = [d for d in path.glob('*') if d.is_dir()]

    file_list = []  # file path list

    # all extensions / all sub directories
    for ext in extensions:
        for _dir in dirs:
            glob = _dir.glob(f'*.{ext}')
            tmp = [f.as_posix() for f in glob if f.is_file()]
            file_list.extend(tmp)

    return file_list, dirs
