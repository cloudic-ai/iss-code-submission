from datetime import datetime, timedelta
from threading import Thread
from camera import get_image
from exceptions import CameraNotAvailable
from image_processing import compress
from constants import MAX_EXECUTION_TIME
from helpers import make_sure_path_exists, data_folder, tmp_folder
from setup_logging import setup_logging, get_logger

start_time = datetime.now()

# Set up logging
setup_logging(start_time)
logger = get_logger(__name__)

logger.info("Starting program")

alive = True

make_sure_path_exists(data_folder)
make_sure_path_exists(tmp_folder)

while datetime.now() - start_time < timedelta(seconds=MAX_EXECUTION_TIME) and alive:
    try:
        # The taking and processing of images is done in separate threads
        logger.info("Starting threads")
        camera_thread = Thread(target=get_image, args=[start_time])
        cloud_detection_thread = Thread(
            target=compress, args=[start_time])
        camera_thread.start()
        cloud_detection_thread.start()
        camera_thread.join()
        cloud_detection_thread.join()
        logger.info("Threads finished")
        alive = False
    except CameraNotAvailable:
        logger.info(
            "The program cannot operate without a camera and will now shut down")
        alive = False
    except (KeyboardInterrupt, SystemExit):
        logger.info("KeyboardInterrupt or SystemExit")
        alive = False
    except BaseException as e:
        logger.error(e)
        logger.info(
            "The program seems to have crashed. If there is enough time left, it will now restart.")

end_time = datetime.now()
logger.info("Program finished")
logger.info("Total time: %s", end_time - start_time)
