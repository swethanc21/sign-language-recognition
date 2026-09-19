from sign_language_detection.entity.config_entity import DataValidationConfig
from sign_language_detection.logging.logger import logger
from sign_language_detection.exception.exception import CustomException

import sys
import pandas as pd


class DataValidation:

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
                "All processed CSV files loaded"
            )

            return (
                train_data,
                val_data,
                test_data
            )

        except Exception as e:

            logger.error(
                "Failed to load processed CSV files"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


    def data_sanity(self):

        try:

            train_data, val_data, test_data = (
                self._load_prepared_data()
            )

            expected_columns = [
                "filename",
                "label"
            ]

            expected_labels = set(
                range(self.config.num_classes)
            )


            datasets = [
                ("train", train_data),
                ("val", val_data),
                ("test", test_data),
            ]


            for split_name, dataframe in datasets:

                if dataframe.empty:

                    raise Exception(
                        f"{split_name} dataset is empty"
                    )

                if list(dataframe.columns) != expected_columns:

                    raise Exception(
                        f"{split_name} columns are invalid"
                    )

                if dataframe["filename"].isna().any():

                    raise Exception(
                        f"{split_name} contains missing filenames"
                    )

                if dataframe["label"].isna().any():

                    raise Exception(
                        f"{split_name} contains missing labels"
                    )

                if dataframe["filename"].duplicated().any():

                    raise Exception(
                        f"{split_name} contains duplicate filenames"
                    )

                if not pd.api.types.is_integer_dtype(
                    dataframe["label"]
                ):

                    raise Exception(
                        f"{split_name} labels must be integer"
                    )

                if set(dataframe["label"]) != expected_labels:

                    raise Exception(
                        f"{split_name} labels are invalid"
                    )

                if not dataframe["filename"].apply(
                    lambda x: isinstance(x, str)
                ).all():

                    raise Exception(
                        f"{split_name} filenames must be strings"
                    )

                if dataframe["filename"].str.strip().ne(
                    dataframe["filename"]
                ).any():

                    raise Exception(
                        f"{split_name} filenames contain whitespace"
                    )

                if not dataframe["filename"].str.lower().str.endswith(
                    ".mp4"
                ).all():

                    raise Exception(
                        f"{split_name} contains invalid video files"
                    )


            train_files = set(
                train_data["filename"]
            )

            val_files = set(
                val_data["filename"]
            )

            test_files = set(
                test_data["filename"]
            )


            if train_files.intersection(val_files):

                raise Exception(
                    "Data leakage between train and validation"
                )

            if train_files.intersection(test_files):

                raise Exception(
                    "Data leakage between train and test"
                )

            if val_files.intersection(test_files):

                raise Exception(
                    "Data leakage between validation and test"
                )


            logger.info(
                "Data sanity checks completed successfully"
            )

            return True

        except Exception as e:

            logger.error(
                "Data sanity check failed"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


    def initiate_data_validation(self):

        try:

            result = self.data_sanity()

            logger.info(
                "Data validation completed successfully"
            )

            return result

        except Exception as e:

            logger.error(
                "Data validation failed"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )