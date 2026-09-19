from sign_language_detection.entity.config_entity import DataTransformationConfig
from sign_language_detection.logging.logger import logger
from sign_language_detection.exception.exception import CustomException

import sys
import pandas as pd


class DataTransformation:

    def __init__(self, config):

        self.config = config


    def _load_prepared_data(self):

        try:

            paths = [
                self.config.train_processed_csv,
                self.config.val_processed_csv,
                self.config.test_processed_csv,
            ]

            for path in paths:

                if not path.exists():

                    raise Exception(
                        f"Processed CSV does not exist: {path}"
                    )

            train_data = pd.read_csv(
                self.config.train_processed_csv
            )

            val_data = pd.read_csv(
                self.config.val_processed_csv
            )

            test_data = pd.read_csv(
                self.config.test_processed_csv
            )

            logger.info(
                "Prepared metadata loaded successfully"
            )

            return (
                train_data,
                val_data,
                test_data
            )

        except Exception as e:

            logger.error(
                "Failed to load prepared metadata"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


    def _validate_video_directories(self):

        directories = [
            self.config.train_video_dir,
            self.config.val_video_dir,
            self.config.test_video_dir,
        ]

        for directory in directories:

            if not directory.exists():

                raise Exception(
                    f"Video directory does not exist: {directory}"
                )


    def initiate_data_transformation(self):

        try:

            train_data, val_data, test_data = (
                self._load_prepared_data()
            )

            self._validate_video_directories()


            logger.info(
                f"Train videos: {len(train_data)}"
            )

            logger.info(
                f"Validation videos: {len(val_data)}"
            )

            logger.info(
                f"Test videos: {len(test_data)}"
            )

            logger.info(
                f"Temporal frames: "
                f"{self.config.num_frames}"
            )

            logger.info(
                f"Image size: "
                f"{self.config.image_height}x"
                f"{self.config.image_width}"
            )

            logger.info(
                "Dynamic video transformation will be "
                "performed inside the Dataset."
            )

            logger.info(
                "Data transformation configuration "
                "completed successfully."
            )

            return True

        except Exception as e:

            logger.error(
                "Data transformation failed"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )