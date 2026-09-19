from sign_language_detection.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from sign_language_detection.constant.constants import (
    DATA_DIR,
    ARTIFACTS_DIR,

    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,

    TRAIN_VIDEO_DIR,
    VAL_VIDEO_DIR,
    TEST_VIDEO_DIR,

    DEV_SAMPLES_PER_CLASS,
    DEV_VAL_SAMPLES_PER_CLASS,

    NUM_CLASSES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_FRAMES,

    BATCH_SIZE,
    EPOCHS,

    GRAD_ACCUM_STEPS,
    NUM_WORKERS,

    LAYER3_LEARNING_RATE,
    LAYER4_LEARNING_RATE,
    FC_LEARNING_RATE,

    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    GRADIENT_CLIP_VALUE,

    SCHEDULER_T0,
    SCHEDULER_T_MULT,
    SCHEDULER_ETA_MIN,

    DROPOUT_RATE,

    EARLY_STOPPING_PATIENCE,

    TRAIN_PROCESSED_CSV,
    VAL_PROCESSED_CSV,
    TEST_PROCESSED_CSV,

    MODEL_DIR,
    MODEL_FILE,

    CHECKPOINT_DIR,
    CHECKPOINT_FILE,
)


class ConfigurationManager:

    def __init__(self):

        self.data_dir = DATA_DIR

        self.artifacts_dir = ARTIFACTS_DIR


    def get_data_ingestion_config(self):

        config = DataIngestionConfig(

            data_dir=self.data_dir,

            artifacts_dir=self.artifacts_dir,

            train_csv=TRAIN_CSV,
            val_csv=VAL_CSV,
            test_csv=TEST_CSV,

            train_processed_csv=TRAIN_PROCESSED_CSV,
            val_processed_csv=VAL_PROCESSED_CSV,
            test_processed_csv=TEST_PROCESSED_CSV,

            train_video_dir=TRAIN_VIDEO_DIR,
            val_video_dir=VAL_VIDEO_DIR,
            test_video_dir=TEST_VIDEO_DIR,

            num_classes=NUM_CLASSES,

            dev_samples_per_class=DEV_SAMPLES_PER_CLASS,
            dev_val_samples_per_class=DEV_VAL_SAMPLES_PER_CLASS,
        )

        return config


    def get_data_validation_config(self):

        config = DataValidationConfig(

            train_processed_csv=TRAIN_PROCESSED_CSV,
            val_processed_csv=VAL_PROCESSED_CSV,
            test_processed_csv=TEST_PROCESSED_CSV,

            num_classes=NUM_CLASSES,
        )

        return config


    def get_data_transformation_config(self):

        config = DataTransformationConfig(

            train_processed_csv=TRAIN_PROCESSED_CSV,
            val_processed_csv=VAL_PROCESSED_CSV,
            test_processed_csv=TEST_PROCESSED_CSV,

            num_frames=NUM_FRAMES,

            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,

            train_video_dir=TRAIN_VIDEO_DIR,
            val_video_dir=VAL_VIDEO_DIR,
            test_video_dir=TEST_VIDEO_DIR,
        )

        return config


    def get_model_trainer_config(self):

        config = ModelTrainerConfig(

            num_classes=NUM_CLASSES,

            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,

            num_frames=NUM_FRAMES,

            batch_size=BATCH_SIZE,

            epochs=EPOCHS,

            grad_accum_steps=GRAD_ACCUM_STEPS,

            num_workers=NUM_WORKERS,

            layer3_learning_rate=LAYER3_LEARNING_RATE,
            layer4_learning_rate=LAYER4_LEARNING_RATE,
            fc_learning_rate=FC_LEARNING_RATE,

            weight_decay=WEIGHT_DECAY,

            label_smoothing=LABEL_SMOOTHING,

            gradient_clip_value=GRADIENT_CLIP_VALUE,

            scheduler_t0=SCHEDULER_T0,
            scheduler_t_mult=SCHEDULER_T_MULT,
            scheduler_eta_min=SCHEDULER_ETA_MIN,

            dropout_rate=DROPOUT_RATE,

            early_stopping_patience=EARLY_STOPPING_PATIENCE,

            train_processed_csv=TRAIN_PROCESSED_CSV,
            val_processed_csv=VAL_PROCESSED_CSV,
            test_processed_csv=TEST_PROCESSED_CSV,

            train_video_dir=TRAIN_VIDEO_DIR,
            val_video_dir=VAL_VIDEO_DIR,
            test_video_dir=TEST_VIDEO_DIR,

            model_dir=MODEL_DIR,
            model_file=MODEL_FILE,

            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_file=CHECKPOINT_FILE,
        )

        return config