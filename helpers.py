from os import makedirs, path, walk


def get_path_size(start_path: str):
    total_size = 0
    for dirpath, dirnames, filenames in walk(start_path):
        for f in filenames:
            fp = path.join(dirpath, f)
            total_size += path.getsize(fp)
    return total_size


def make_sure_path_exists(_path: str):
    if not path.exists(_path):
        makedirs(_path)
