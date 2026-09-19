from sign_language_detection.configuration import ConfigurationManager

from sign_language_detection.components.data_ingestion import (
    DataIngestion
)

from sign_language_detection.components.data_validation import (
    DataValidation
)

from sign_language_detection.components.data_transformation import (
    DataTransformation
)

from sign_language_detection.components.model_trainer import (
    ModelTrainer
)


def main():

    config_manager = ConfigurationManager()

    # =========================================================
    # 1. DATA INGESTION
    # =========================================================

    data_ingestion_config = (
        config_manager.get_data_ingestion_config()
    )

    data_ingestion = DataIngestion(
        config=data_ingestion_config
    )

    (
        train_output,
        val_output,
        test_output
    ) = data_ingestion.initiate_data_ingestion()

    print(
        "\nData Ingestion Completed Successfully"
    )

    print(
        f"Train metadata      : {train_output}"
    )

    print(
        f"Validation metadata : {val_output}"
    )

    print(
        f"Test metadata       : {test_output}"
    )


    # =========================================================
    # 2. DATA VALIDATION
    # =========================================================

    data_validation_config = (
        config_manager.get_data_validation_config()
    )

    data_validation = DataValidation(
        config=data_validation_config
    )

    data_validation_output = (
        data_validation.initiate_data_validation()
    )

    print(
        "\nData Validation Completed Successfully"
    )

    print(
        f"Validation result : "
        f"{data_validation_output}"
    )


    # =========================================================
    # 3. DATA TRANSFORMATION / VIDEO PIPELINE VALIDATION
    # =========================================================

    data_transformation_config = (
        config_manager.get_data_transformation_config()
    )

    data_transformation = DataTransformation(
        config=data_transformation_config
    )

    data_transformation_output = (
        data_transformation.initiate_data_transformation()
    )

    print(
        "\nData Transformation Completed Successfully"
    )

    print(
        f"Transformation result : "
        f"{data_transformation_output}"
    )


    # =========================================================
    # 4. MODEL TRAINING
    # =========================================================

    model_trainer_config = (
        config_manager.get_model_trainer_config()
    )

    model_trainer = ModelTrainer(
        config=model_trainer_config
    )

    model_file = (
        model_trainer.initiate_model_training()
    )

    print(
        "\nModel Training Completed Successfully"
    )

    print(
        f"Model saved at : {model_file}"
    )


if __name__ == "__main__":
    main()