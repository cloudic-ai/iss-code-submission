from datetime import datetime
from logging import Logger, getLogger, basicConfig, DEBUG, INFO
from helpers import make_sure_path_exists, logs_folder


def setup_logging(start_time: datetime):
    make_sure_path_exists(logs_folder)

    basicConfig(filename=f"{logs_folder}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}.log",
                filemode='a',
                datefmt='%H:%M:%S',
                # Format
                # Time (HH:MM:SS:MS) - Level - Subsystem - Message
                format='%(asctime)s.%(msecs)03d: %(levelname)s: %(name)s: %(message)s',
                level=INFO)


def get_logger(subsystem: str) -> Logger:
    return getLogger(subsystem)
