from datetime import datetime
from os import makedirs, path, walk
from constants import MAX_EXECUTION_TIME


def check_time_remaining(start_time: datetime) -> float:
    time_remaining = MAX_EXECUTION_TIME - \
        (datetime.now() - start_time).total_seconds()
    return time_remaining


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
