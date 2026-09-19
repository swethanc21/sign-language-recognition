from sign_language_detection.entity.config_entity import DataIngestionConfig
from sign_language_detection.logging.logger import logger
from sign_language_detection.exception.exception import CustomException

import sys
import pandas as pd


class DataIngestion:

    def __init__(self, config):

        self.config = config


    def helper_development_sampling(
        self,
        dataframe,
        samples_per_class
    ):

        if samples_per_class <= 0:
            return dataframe.copy()

        dataframe_groups = dataframe.groupby("label")

        selected_samples = []

        for label, group in dataframe_groups:

            if group.shape[0] >= samples_per_class:

                selected_samples.append(
                    group.sample(
                        n=samples_per_class,
                        random_state=42
                    )
                )

            else:

                selected_samples.append(
                    group.sample(
                        n=group.shape[0],
                        random_state=42
                    )
                )

        return pd.concat(
            selected_samples,
            ignore_index=True
        )


    def _check_video_availability(
        self,
        dataframe,
        video_dir
    ):

        available_samples = []

        missing_files = []

        for _, row in dataframe.iterrows():

            filename = row["filename"]

            video_path = video_dir / filename

            if video_path.exists():

                available_samples.append(row)

            else:

                missing_files.append(row)

                logger.warning(
                    f"Video path not found: {video_path}"
                )

        available_data = pd.DataFrame(
            available_samples
        )

        return available_data, missing_files


    def _save_prepared_metadata(
        self,
        train_data,
        val_data,
        test_data
    ):

        try:

            train_output = (
                self.config.train_processed_csv
            )

            val_output = (
                self.config.val_processed_csv
            )

            test_output = (
                self.config.test_processed_csv
            )

            train_output.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            train_data.to_csv(
                train_output,
                index=False
            )

            val_data.to_csv(
                val_output,
                index=False
            )

            test_data.to_csv(
                test_output,
                index=False
            )

            logger.info(
                "Prepared metadata saved successfully"
            )

            return (
                train_output,
                val_output,
                test_output
            )

        except Exception as e:

            logger.error(
                "Failed to save prepared metadata"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


    def _check_raw_data(self):

        try:

            train_csv = self.config.train_csv
            val_csv = self.config.val_csv
            test_csv = self.config.test_csv

            if not train_csv.exists():
                raise Exception(
                    "Train CSV not found"
                )

            if not val_csv.exists():
                raise Exception(
                    "Validation CSV not found"
                )

            if not test_csv.exists():
                raise Exception(
                    "Test CSV not found"
                )

            logger.info(
                "All CSV files found"
            )


            train_video_dir = (
                self.config.train_video_dir
            )

            val_video_dir = (
                self.config.val_video_dir
            )

            test_video_dir = (
                self.config.test_video_dir
            )

            if not train_video_dir.exists():

                raise Exception(
                    f"Train video directory not found: "
                    f"{train_video_dir}"
                )

            if not val_video_dir.exists():

                raise Exception(
                    f"Validation video directory not found: "
                    f"{val_video_dir}"
                )

            if not test_video_dir.exists():

                raise Exception(
                    f"Test video directory not found: "
                    f"{test_video_dir}"
                )

            logger.info(
                "Video directories are ready"
            )

        except Exception as e:

            logger.error(
                "Raw data check failed"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


    def prepare_data(self):

        try:

            train_data = pd.read_csv(
                self.config.train_csv,
                header=None,
                names=["filename", "label"]
            )

            val_data = pd.read_csv(
                self.config.val_csv,
                header=None,
                names=["filename", "label"]
            )

            test_data = pd.read_csv(
                self.config.test_csv,
                header=None,
                names=["filename", "label"]
            )

            logger.info(
                "CSV files loaded successfully"
            )

        except Exception as e:

            logger.error(
                "CSV files were not loaded successfully"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


        try:

            expected_num_classes = (
                self.config.num_classes
            )

            expected_labels = set(
                range(expected_num_classes)
            )

            datasets = [
                ("train", train_data),
                ("val", val_data),
                ("test", test_data),
            ]

            for split_name, dataframe in datasets:

                if dataframe.empty:

                    raise Exception(
                        f"{split_name} data is empty"
                    )

                if list(dataframe.columns) != [
                    "filename",
                    "label"
                ]:

                    raise Exception(
                        f"{split_name} data columns are invalid"
                    )

                if dataframe["filename"].isna().any():

                    raise Exception(
                        f"{split_name} filenames contain "
                        f"missing values"
                    )

                if dataframe["label"].isna().any():

                    raise Exception(
                        f"{split_name} labels contain "
                        f"missing values"
                    )

                if dataframe["filename"].duplicated().any():

                    raise Exception(
                        f"{split_name} contains duplicate filenames"
                    )

                if set(dataframe["label"]) != expected_labels:

                    raise Exception(
                        f"{split_name} labels are invalid"
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

        except Exception as e:

            logger.error(
                "Data sanity check failed"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


        try:

            dev_train_data = (
                self.helper_development_sampling(
                    train_data,
                    self.config.dev_samples_per_class
                )
            )

            dev_val_data = (
                self.helper_development_sampling(
                    val_data,
                    self.config.dev_val_samples_per_class
                )
            )

            logger.info(
                f"Prepared train data shape: "
                f"{dev_train_data.shape}"
            )

            logger.info(
                f"Prepared validation data shape: "
                f"{dev_val_data.shape}"
            )

        except Exception as e:

            logger.error(
                "Development sampling failed"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


        try:

            available_train_data, missing_train_files = (
                self._check_video_availability(
                    dev_train_data,
                    self.config.train_video_dir
                )
            )

            available_val_data, missing_val_files = (
                self._check_video_availability(
                    dev_val_data,
                    self.config.val_video_dir
                )
            )

            available_test_data, missing_test_files = (
                self._check_video_availability(
                    test_data,
                    self.config.test_video_dir
                )
            )


            if available_train_data.empty:

                raise Exception(
                    "No training videos are available"
                )

            if available_val_data.empty:

                raise Exception(
                    "No validation videos are available"
                )

            if available_test_data.empty:

                raise Exception(
                    "No testing videos are available"
                )


            logger.info(
                f"Available train videos: "
                f"{len(available_train_data)}"
            )

            logger.info(
                f"Missing train videos: "
                f"{len(missing_train_files)}"
            )

            logger.info(
                f"Available val videos: "
                f"{len(available_val_data)}"
            )

            logger.info(
                f"Missing val videos: "
                f"{len(missing_val_files)}"
            )

            logger.info(
                f"Available test videos: "
                f"{len(available_test_data)}"
            )

            logger.info(
                f"Missing test videos: "
                f"{len(missing_test_files)}"
            )

        except Exception as e:

            logger.error(
                "Video availability check failed"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


        return self._save_prepared_metadata(
            available_train_data,
            available_val_data,
            available_test_data
        )


    def initiate_data_ingestion(self):

        try:

            self._check_raw_data()

            outputs = self.prepare_data()

            logger.info(
                "Data ingestion completed successfully"
            )

            return outputs

        except Exception as e:

            logger.error(
                "Data ingestion failed"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )