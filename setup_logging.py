from datetime import datetime
from logging import Logger, getLogger, basicConfig, DEBUG, INFO
from helpers import make_sure_path_exists


def setup_logging(start_time: datetime):
    make_sure_path_exists('logs')

    basicConfig(filename='logs/' + start_time.strftime("%Y-%m-%d_%H-%M-%S") + '.log',
                filemode='a',
                datefmt='%H:%M:%S',
                level=INFO)


def get_logger(subsystem: str) -> Logger:
    return getLogger('deecream.' + subsystem)
