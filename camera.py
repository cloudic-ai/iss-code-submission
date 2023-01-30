from os import listdir, makedirs, path
from time import sleep
from random import choice
from cv2 import cvtColor, imread, imwrite, COLOR_BGR2RGB
from exceptions import DataFolderFull, ExecutionTimeExceeded
from setup_logging import get_logger
from datetime import datetime
from constants import MAX_EXECUTION_TIME, MAX_SIZE_DATA
from helpers import get_path_size
from math import floor

logger = get_logger(__name__)


def check_time_remaining(start_time: datetime) -> float:
    time_remaining = MAX_EXECUTION_TIME - \
        (datetime.now() - start_time).total_seconds()
    logger.info(f"Time remaining: {time_remaining}")
    return time_remaining


def check_data_folder_space_remaining() -> int:
    data_folder_space_remaining = MAX_SIZE_DATA - get_path_size("data")
    logger.info(
        f"Space remaining in data folder: {data_folder_space_remaining}")
    return data_folder_space_remaining


def calculate_sleep_time(start_time: datetime, average_image_size: int):
    time_remaining = check_time_remaining(start_time)

    data_folder_space_remaining = check_data_folder_space_remaining()
    images_remaining = floor(data_folder_space_remaining /
                             average_image_size)
    logger.info(f"Images remaining: {images_remaining}")

    if images_remaining <= 0:
        raise DataFolderFull(average_image_size)

    sleep_time = (time_remaining / images_remaining)
    logger.info(f"Calculated sleep time: {sleep_time}")

    if time_remaining < 0:
        raise ExecutionTimeExceeded

    if sleep_time > time_remaining:
        sleep_time = 0

    return sleep_time


def get_image(start_time: datetime):
    """
    Get image from camera and save it to the data folder. Sleep until the data folder has enough space for the next image.

    Notes:
    - The function will save at most one image per second.
    - The function might exceed the MAX_DATA_FOLDER_SIZE by up to the size of one image.
    """
    sum_image_sizes = 0
    image_count = 0

    while check_time_remaining(start_time) > 0:
        try:
            # Wait until there is any space in the data folder
            while check_data_folder_space_remaining() <= 0:
                logger.info("Waiting for data folder to have space")
                if check_time_remaining(start_time) <= 1:
                    raise ExecutionTimeExceeded
                sleep(1)

            # Get image from camera
            file_name = choice(listdir("debug-images"))
            image = cvtColor(
                imread(f"debug-images/{file_name}"), COLOR_BGR2RGB)
            logger.info(f"Image loaded from 'debug-images/{file_name}'")

            time = datetime.now()

            # Save image to data folder
            makedirs("data/" + time.strftime("%Y-%m-%d_%H-%M-%S"))
            imwrite(
                f"data/{time.strftime('%Y-%m-%d_%H-%M-%S')}/camera.jpg", image)
            logger.info(
                f"Image saved to 'data/{time.strftime('%Y-%m-%d_%H-%M-%S')}/camera.jpg'")

            # Update the average image size
            size_of_image = path.getsize(
                f"data/{time.strftime('%Y-%m-%d_%H-%M-%S')}/camera.jpg")
            logger.info(f"Size of last saved image: {size_of_image}")
            sum_image_sizes += size_of_image
            image_count += 1
            average_image_size = sum_image_sizes / image_count

            # Wait until data folder has enough space for the next image
            while floor(check_data_folder_space_remaining() / average_image_size) < 1:
                logger.info("Waiting for data folder to have enough space")
                if check_time_remaining(start_time) <= 1:
                    raise ExecutionTimeExceeded
                sleep(1)

            # Calculate sleep time
            sleep_time = calculate_sleep_time(
                start_time, int(average_image_size))

            # Sleep
            sleep(max(sleep_time, 1))
        except ExecutionTimeExceeded:
            logger.info("Execution time exceeded")
            break
        except Exception as e:
            logger.exception(e)
