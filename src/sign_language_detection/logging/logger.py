import logging
from datetime import datetime
from sign_language_detection.constant.constants import PROJECT_ROOT


LOG_DIR=PROJECT_ROOT/"logs"
LOG_DIR.mkdir(parents=True,exist_ok=True)
now=datetime.now()
filename=now.strftime("%Y_%m_%d_%H_%M_%S.log")
LOG_FILE=LOG_DIR/filename

logger = logging.getLogger("sign_language_detection")
logger.setLevel(logging.INFO)

file_handler=logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)

formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.info("Logging system started")