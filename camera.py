from os import makedirs, path, remove
from time import sleep
from cv2 import Mat, imread, imwrite
from exceptions import CameraNotAvailable, DataFolderFull, ExecutionTimeExceeded
from setup_logging import get_logger
from datetime import datetime
from constants import MAX_SIZE_DATA, ORIGINAL_IMAGE_NAME
from helpers import check_time_remaining, get_path_size, data_folder, tmp_folder
from math import floor

logger = get_logger(__name__)


def check_data_folder_space_remaining() -> int:
    data_folder_space_remaining = MAX_SIZE_DATA - get_path_size(data_folder)
    logger.debug(
        f"Space remaining in data folder: {data_folder_space_remaining}")
    return data_folder_space_remaining


def calculate_sleep_time(start_time: datetime, average_image_size: int) -> float:
    time_remaining = check_time_remaining(start_time)

    data_folder_space_remaining = check_data_folder_space_remaining()
    images_remaining = floor(data_folder_space_remaining /
                             average_image_size)
    logger.debug(f"Images remaining: {images_remaining}")

    if images_remaining <= 0:
        raise DataFolderFull(average_image_size)

    # Calculate sleep time (in seconds) but at least 1 second
    sleep_time = max((time_remaining / images_remaining), 1)
    logger.debug(f"Calculated sleep time: {sleep_time}")

    if time_remaining < 0:
        raise ExecutionTimeExceeded

    if sleep_time > time_remaining:
        sleep_time = 0.0

    return sleep_time


def is_night_image(cropped_image: Mat) -> bool:
    # Calculate average brightness
    avg_brightness = cropped_image.mean()
    # Calculate brightest pixel value
    max_brightness = cropped_image.max()

    return avg_brightness < 60 or max_brightness < 200


def take_picture(camera) -> Mat:
    time = datetime.now()
    image_path = f"{tmp_folder}/{time.strftime('%Y-%m-%d_%H-%M-%S-%f')}.jpg"

    # Take picture
    camera.capture(image_path)

    # Load image
    image = imread(image_path)

    # Delete image
    remove(image_path)

    return image


def get_image(start_time: datetime) -> None:
    """
    Get image from camera and save it to the data folder. Sleep until the data folder has enough space for the next image.

    Notes:
    - The function will save at most one image per second.
    - The function might exceed the MAX_DATA_FOLDER_SIZE by up to the size of one image.
    """
    # Initialize camera
    try:
        from picamera import PiCamera
        camera = PiCamera()
    except Exception as e:
        # In case the camera is not available, st
        logger.error(f"Error while importing PiCamera: {e}")
        logger.info(
            "The program will continue without the camera thread")
        return

    sum_image_sizes = 0
    image_count = 0

    while check_time_remaining(start_time) > 0:
        try:
            # Wait until there is any space in the data folder
            while check_data_folder_space_remaining() <= 0:
                logger.debug("Waiting for data folder to have space")
                if check_time_remaining(start_time) <= 1:
                    raise ExecutionTimeExceeded
                sleep(1)

            # Get image
            image = take_picture(camera)

            # Take a square with the edge length equal to the height of the image from the center of the image
            y1 = 0
            y2 = image.shape[0]
            x1 = int((image.shape[1] - image.shape[0]) / 2)
            x2 = x1 + image.shape[0]
            image_cropped = image[y1:y2, x1:x2]

            time = datetime.now()

            # Check if image is a night image and skip if it is
            if is_night_image(image_cropped):
                logger.info("Night image detected")
                sleep(1)
                continue

            # Save image to data folder
            folder_name = f"{data_folder}/{time.strftime('%Y-%m-%d_%H-%M-%S-%f')}"
            makedirs(folder_name)
            imwrite(
                f"{folder_name}/{ORIGINAL_IMAGE_NAME}", image_cropped)
            logger.info(
                f"Image saved to '{folder_name}/{ORIGINAL_IMAGE_NAME}'")

            # Update the average image size
            size_of_image = path.getsize(
                f"{folder_name}/{ORIGINAL_IMAGE_NAME}")
            logger.debug(f"Size of last saved image: {size_of_image}")
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
            sleep(sleep_time)
        except ExecutionTimeExceeded:
            logger.info("Execution time exceeded")
            break
        except Exception as e:
            logger.error(f"Error while getting image: {e}")

    logger.info(
        f"Camera thread finished: {image_count} images saved, {sum_image_sizes} bytes total")
