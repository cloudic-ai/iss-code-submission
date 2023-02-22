from datetime import datetime
from os import listdir, path, remove
from time import sleep
from cv2 import COLOR_GRAY2RGB, INTER_AREA, Mat, cvtColor, imread, imwrite, resize
from constants import MASKED_IMAGE_NAME, ORIGINAL_IMAGE_NAME
from helpers import check_time_remaining, data_folder
import tensorflow as tf
import numpy as np
from models.AE import AE, BinaryFN, BinaryFP, BinaryTN, BinaryTP
from setup_logging import get_logger


logger = get_logger(__name__)


def create_cloud_mask(image: Mat, model: AE) -> Mat:
    logger.info("Creating cloud mask")

    resized = resize(image, (256, 256), interpolation=INTER_AREA)

    input_image = np.reshape(
        resized, (1, resized.shape[0], resized.shape[1], 3))

    result = model.predict(input_image)

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
        model = AE()
        model.build((None, 256, 256, 3))
        model.load_weights("models/checkpoint-ae.h5")
        optimizer = tf.keras.optimizers.get(
            {"class_name": "Adam", "config": {"learning_rate": 1e-3}})
        loss_fn = tf.keras.losses.get("MeanSquaredError")
        model.compile(loss=loss_fn,
                      optimizer=optimizer,
                      metrics=[BinaryTP(),
                               BinaryFP(),
                               BinaryTN(),
                               BinaryFN()])
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
                    cloud_mask = create_cloud_mask(image, model)

                    # Scale values from 0 to 255 (because result of model is between 0 and 1)
                    # cloud_mask_scaled = cloud_mask * 255  # type: ignore

                    # Save cloud mask
                    # logger.info(f"Saving cloud mask for '{folder}/{ORIGINAL_IMAGE_NAME}'")
                    # imwrite(f"{data_folder}/{folder}/{MASK_NAME}", cvtColor(cloud_mask_scaled, COLOR_GRAY2RGB))

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
