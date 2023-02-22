from datetime import datetime
from os import listdir, path, remove
from time import sleep
from cv2 import COLOR_GRAY2RGB, INTER_AREA, Mat, cvtColor, imread, imwrite, resize
from constants import MASKED_IMAGE_NAME, ORIGINAL_IMAGE_NAME
from helpers import check_time_remaining, data_folder, models_folder
import tensorflow.lite as tf_lite
import numpy as np
from setup_logging import get_logger


logger = get_logger(__name__)


def create_cloud_mask(image: Mat, interpreter) -> Mat:
    logger.info("Creating cloud mask")

    resized = resize(image, (256, 256), interpolation=INTER_AREA)

    input_image = np.reshape(
        resized, (1, resized.shape[0], resized.shape[1], 3))

    result = interpreter.run(input_image)[0]

    output_image = np.reshape(result, (resized.shape[0], resized.shape[1], 1))

    return output_image


def apply_cloud_mask(image: Mat, cloud_mask: Mat) -> Mat:
    logger.info("Applying cloud mask")

    # Scale mask to size of image
    cloud_mask_scaled = resize(
        cloud_mask, (image.shape[0], image.shape[1]), interpolation=INTER_AREA)
    cloud_mask_scaled = cvtColor(cloud_mask_scaled, COLOR_GRAY2RGB)

    # Apply cloud mask
    masked_image = np.multiply(image, cloud_mask_scaled)

    return masked_image


def compress(start_time: datetime) -> None:
    # Initialize model
    try:
        from PIL import Image
        from pycoral.utils.edgetpu import make_interpreter

        interpreter = make_interpreter(
            f"{models_folder}/ae_tf_lite_model.tflite")
        interpreter.allocate_tensors()
    except Exception as e:
        # In case the model fails to load, shut down the image processing thread (other threads will continue)
        logger.error(f"Error while loading model: {e}")
        logger.info(
            "The program will continue without the image processing thread")
        return
    logger.info("Model initialized")

    # Get image directories in data folder (excluding the last one, because it might be incomplete)
    image_folders = sorted(listdir(data_folder))[:-1]
    processed_images = []
    skipped_images = []

    while check_time_remaining(start_time) > 0:
        for folder in image_folders:
            if folder not in processed_images and folder not in skipped_images:
                if path.isfile(f"{data_folder}/{folder}/{MASKED_IMAGE_NAME}") or not path.isfile(f"{data_folder}/{folder}/{ORIGINAL_IMAGE_NAME}"):
                    logger.info(
                        f"Skipping image '{folder}/{ORIGINAL_IMAGE_NAME}' because it seems to have already been processed")
                    skipped_images.append(folder)
                    continue

                while not path.getsize(f"{data_folder}/{folder}/{ORIGINAL_IMAGE_NAME}") > 0 or imread(f"{data_folder}/{folder}/{ORIGINAL_IMAGE_NAME}") is None:
                    logger.info(
                        f"Waiting for image '{data_folder}/{folder}/{ORIGINAL_IMAGE_NAME}' to be fully written to disk")
                    sleep(1)

                try:
                    logger.info(
                        f"Processing image '{folder}/{ORIGINAL_IMAGE_NAME}'")

                    # Load image
                    image = imread(
                        f"{data_folder}/{folder}/{ORIGINAL_IMAGE_NAME}")

                    # Create cloud mask
                    cloud_mask = create_cloud_mask(image, interpreter)

                    # Apply cloud mask
                    image_masked = apply_cloud_mask(image, cloud_mask)
                    imwrite(
                        f"{data_folder}/{folder}/{MASKED_IMAGE_NAME}", image_masked)

                    # Delete original image
                    logger.info(
                        f"Deleting original image '{folder}/{ORIGINAL_IMAGE_NAME}'")
                    remove(f"{data_folder}/{folder}/{ORIGINAL_IMAGE_NAME}")
                except Exception as e:
                    # In case the cloud mask fails to create, skip this image
                    logger.error(
                        f"Error while processing image '{folder}/{ORIGINAL_IMAGE_NAME}': {e}")
                    logger.info(
                        f"Skipping image '{folder}/{ORIGINAL_IMAGE_NAME}'")
                    skipped_images.append(folder)
                    continue

                processed_images.append(folder)

        # Check if new images have been added
        new_image_folders = sorted(listdir(data_folder))[:-1]
        for folder in new_image_folders:
            if folder not in image_folders:
                image_folders.append(folder)

    logger.info(
        f"Image processing thread finished: {len(processed_images)} images processed, {len(skipped_images)} images skipped")
