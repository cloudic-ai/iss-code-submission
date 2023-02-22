from datetime import datetime, timedelta
from threading import Thread
from camera import get_image
from image_processing import compress
from constants import MAX_EXECUTION_TIME
from helpers import make_sure_path_exists, data_folder
from setup_logging import setup_logging, get_logger

start_time = datetime.now()

# Set up logging
setup_logging(start_time)
logger = get_logger(__name__)

logger.info("Starting at %s", start_time)

alive = True

make_sure_path_exists(data_folder)

while datetime.now() - start_time < timedelta(seconds=MAX_EXECUTION_TIME) and alive:
    try:
        camera_thread = Thread(target=get_image, args=[start_time])
        cloud_detection_thread = Thread(
            target=compress, args=[start_time])
        camera_thread.start()
        cloud_detection_thread.start()
        camera_thread.join()
        cloud_detection_thread.join()
        alive = False
    except (KeyboardInterrupt, SystemExit):
        logger.info("KeyboardInterrupt or SystemExit")
        alive = False
    except BaseException as e:
        logger.exception(e)

end_time = datetime.now()
logger.info("Ending at %s", end_time)
logger.info("Total time: %s", end_time - start_time)
