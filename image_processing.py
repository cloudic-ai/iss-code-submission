from datetime import datetime
from os import listdir, path
from time import sleep
from cv2 import COLOR_GRAY2RGB, INTER_AREA, Mat, cvtColor, imread, imwrite, resize
from helpers import check_time_remaining
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

    # Get directories in Data folder (excluding the last one, because that one might be incomplete)
    image_folders = listdir("data")[:-1]
    processed_images = []
    skipped_images = []

    while check_time_remaining(start_time) > 0:
        for folder in image_folders:
            if folder not in processed_images and folder not in skipped_images:
                if path.isfile(f"data/{folder}/cloud_mask.jpg"):
                    logger.info(
                        f"Skipping image '{folder}/camera.jpg' because cloud mask already exists")
                    skipped_images.append(folder)
                    continue

                # Wait for image to be fully written to disk
                while not path.isfile(f"data/{folder}/camera.jpg") or path.getsize(f"data/{folder}/camera.jpg") == 0:
                    sleep(1)

                try:
                    logger.info(f"Processing image '{folder}/camera.jpg'")

                    # Load image
                    image = imread(f"data/{folder}/camera.jpg")

                    # Create cloud mask
                    cloud_mask = create_cloud_mask(image, model)
                    # Scale values from 0 to 255 (because result of model is between 0 and 1)
                    cloud_mask_scaled = cloud_mask * 255  # type: ignore

                    # Save cloud mask
                    logger.info(f"Saving cloud mask for '{folder}/camera.jpg'")
                    imwrite(f"data/{folder}/cloud_mask.jpg", cvtColor(
                        cloud_mask_scaled, COLOR_GRAY2RGB))
                except Exception as e:
                    # In case the cloud mask fails to create, skip this image
                    logger.error(
                        f"Error while processing image '{folder}/camera.jpg': {e}")
                    logger.info(f"Skipping image '{folder}/camera.jpg'")
                    skipped_images.append(folder)
                    continue

                processed_images.append(folder)

        # Check if new images have been added
        new_image_folders = listdir("data")[:-1]
        for folder in new_image_folders:
            if folder not in image_folders:
                image_folders.append(folder)

    logger.info(
        f"Image processing thread finished: {len(processed_images)} images processed, {len(skipped_images)} images skipped")
