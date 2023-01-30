from datetime import datetime
import logging
from os import makedirs, path

from helpers import make_sure_path_exists


def setup_logging(start_time: datetime):
    make_sure_path_exists('logs')

    logging.basicConfig(filename='logs/' + start_time.strftime("%Y-%m-%d_%H-%M-%S") + '.log',
                        filemode='a',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)


def get_logger(subsystem: str) -> logging.Logger:
    return logging.getLogger('deecream.' + subsystem)
