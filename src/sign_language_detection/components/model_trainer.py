from sign_language_detection.entity.config_entity import ModelTrainerConfig
from sign_language_detection.logging.logger import logger
from sign_language_detection.exception.exception import CustomException

import sys
import random
import time

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights

from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# Kinetics-400 Normalization
# ============================================================

KINETICS_MEAN = np.array(
    [0.43216, 0.394666, 0.37645],
    dtype=np.float32
)

KINETICS_STD = np.array(
    [0.22803, 0.22145, 0.21699],
    dtype=np.float32
)


# ============================================================
# Video Utilities
# ============================================================

def make_black_clip(
    num_frames=8,
    image_size=160
):
    """
    Create a black fallback clip when video decoding fails.

    Output:
        Tensor shape = [3, T, H, W]
    """

    frames = np.zeros(
        (
            num_frames,
            image_size,
            image_size,
            3
        ),
        dtype=np.float32
    )

    frames = (
        frames - KINETICS_MEAN.reshape(1, 1, 1, 3)
    ) / KINETICS_STD.reshape(1, 1, 1, 3)

    frames = torch.from_numpy(
        frames
    )

    frames = frames.permute(
        3,
        0,
        1,
        2
    ).float()

    return frames


def sample_frame_indices(
    total_frames,
    num_frames,
    training=False
):
    """
    Sample frames across the complete temporal extent.

    Training:
        Random frame inside each temporal segment.

    Validation/Test:
        Deterministic center frame of each segment.
    """

    if total_frames <= 0:

        return [0] * num_frames

    boundaries = np.linspace(
        0,
        total_frames,
        num_frames + 1
    ).astype(int)

    indices = []

    for i in range(num_frames):

        start = boundaries[i]

        end = boundaries[i + 1]

        if end <= start:

            idx = min(
                start,
                total_frames - 1
            )

        else:

            if training:

                idx = random.randint(
                    start,
                    end - 1
                )

            else:

                idx = (
                    start + end - 1
                ) // 2

        indices.append(idx)

    return indices


def load_video_clip(
    video_path,
    num_frames=8,
    image_size=160,
    training=False
):
    """
    Load one video and convert it into:

        [3, 8, 160, 160]

    RGB + Kinetics normalization.
    """

    try:

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            cap.release()

            return make_black_clip(
                num_frames=num_frames,
                image_size=image_size
            )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total_frames <= 0:

            cap.release()

            return make_black_clip(
                num_frames=num_frames,
                image_size=image_size
            )

        indices = sample_frame_indices(
            total_frames=total_frames,
            num_frames=num_frames,
            training=training
        )

        frames = []

        last_valid = None

        for idx in indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(idx)
            )

            success, frame = cap.read()

            if (
                not success
                or frame is None
            ):

                if last_valid is None:

                    frame = np.zeros(
                        (
                            image_size,
                            image_size,
                            3
                        ),
                        dtype=np.uint8
                    )

                else:

                    frame = last_valid.copy()

            else:

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                last_valid = frame.copy()

            frame = cv2.resize(
                frame,
                (
                    image_size,
                    image_size
                )
            )

            frames.append(frame)

        cap.release()

        frames = np.stack(
            frames
        ).astype(
            np.float32
        )

        frames = frames / 255.0

        frames = (
            frames
            - KINETICS_MEAN.reshape(1, 1, 1, 3)
        ) / KINETICS_STD.reshape(
            1, 1, 1, 3
        )

        frames = torch.from_numpy(
            frames
        )

        frames = frames.permute(
            3,
            0,
            1,
            2
        ).float()

        return frames

    except Exception:

        return make_black_clip(
            num_frames=num_frames,
            image_size=image_size
        )


# ============================================================
# AUTSL Video Dataset
# ============================================================

class AUTSLVideoDataset(Dataset):

    def __init__(
        self,
        dataframe,
        num_frames=8,
        image_size=160,
        training=False,
        return_metadata=False
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.num_frames = num_frames

        self.image_size = image_size

        self.training = training

        self.return_metadata = (
            return_metadata
        )


    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        video = load_video_clip(
            video_path=row["video_path"],
            num_frames=self.num_frames,
            image_size=self.image_size,
            training=self.training
        )

        label = torch.tensor(
            row["label_idx"],
            dtype=torch.long
        )

        if self.return_metadata:

            return (
                video,
                label,
                idx
            )

        return (
            video,
            label
        )


# ============================================================
# Model Trainer
# ============================================================

class ModelTrainer:

    def __init__(
        self,
        config: ModelTrainerConfig
    ):

        self.config = config

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.use_amp = (
            self.device.type == "cuda"
        )

        self.amp_device = (
            "cuda"
            if self.use_amp
            else "cpu"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp
        )

        logger.info(
            f"Training device: {self.device}"
        )

        if self.device.type == "cuda":

            logger.info(
                f"GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )


    # ========================================================
    # Load Prepared Metadata
    # ========================================================

    def _load_metadata(self):

        try:

            train_csv = (
                self.config.train_processed_csv
            )

            val_csv = (
                self.config.val_processed_csv
            )

            test_csv = (
                self.config.test_processed_csv
            )

            if not train_csv.exists():

                raise Exception(
                    f"Train processed CSV does not exist: "
                    f"{train_csv}"
                )

            if not val_csv.exists():

                raise Exception(
                    f"Validation processed CSV does not exist: "
                    f"{val_csv}"
                )

            if not test_csv.exists():

                raise Exception(
                    f"Test processed CSV does not exist: "
                    f"{test_csv}"
                )

            train_df = pd.read_csv(
                train_csv
            )

            val_df = pd.read_csv(
                val_csv
            )

            test_df = pd.read_csv(
                test_csv
            )

            logger.info(
                f"Train samples: {len(train_df)}"
            )

            logger.info(
                f"Validation samples: {len(val_df)}"
            )

            logger.info(
                f"Test samples: {len(test_df)}"
            )

            return (
                train_df,
                val_df,
                test_df
            )

        except Exception as e:

            logger.error(
                "Failed to load prepared metadata"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


    # ========================================================
    # Add Video Paths
    # ========================================================

    def _prepare_dataframe(
        self,
        dataframe,
        video_dir
    ):

        dataframe = dataframe.copy()

        dataframe["video_path"] = (
            dataframe["filename"]
            .apply(
                lambda x: str(
                    video_dir / x
                )
            )
        )

        dataframe["label_idx"] = (
            dataframe["label"]
            .astype(int)
        )

        missing_count = 0

        for video_path in dataframe[
            "video_path"
        ]:

            if not (
                pd.notna(video_path)
                and
                __import__("pathlib")
                .Path(video_path)
                .exists()
            ):

                missing_count += 1

        if missing_count > 0:

            logger.warning(
                f"{missing_count} video paths "
                f"are missing."
            )

        return dataframe


    # ========================================================
    # Seed Worker
    # ========================================================

    def _seed_worker(
        self,
        worker_id
    ):

        worker_seed = (
            torch.initial_seed()
            % (2 ** 32)
        )

        np.random.seed(
            worker_seed
        )

        random.seed(
            worker_seed
        )


    # ========================================================
    # Create DataLoaders
    # ========================================================

    def _create_dataloaders(self):

        try:

            (
                train_df,
                val_df,
                test_df
            ) = self._load_metadata()


            # ------------------------------------------------
            # Prepare video paths
            # ------------------------------------------------

            train_df = self._prepare_dataframe(
                train_df,
                self.config.train_video_dir
            )

            val_df = self._prepare_dataframe(
                val_df,
                self.config.val_video_dir
            )

            test_df = self._prepare_dataframe(
                test_df,
                self.config.test_video_dir
            )


            # ------------------------------------------------
            # Training dataset
            # Random temporal sampling
            # ------------------------------------------------

            train_dataset = AUTSLVideoDataset(
                train_df,
                num_frames=self.config.num_frames,
                image_size=self.config.image_height,
                training=True
            )


            # ------------------------------------------------
            # Deterministic train evaluation
            # ------------------------------------------------

            train_eval_dataset = AUTSLVideoDataset(
                train_df,
                num_frames=self.config.num_frames,
                image_size=self.config.image_height,
                training=False
            )


            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            val_dataset = AUTSLVideoDataset(
                val_df,
                num_frames=self.config.num_frames,
                image_size=self.config.image_height,
                training=False
            )


            # ------------------------------------------------
            # Test
            # ------------------------------------------------

            test_dataset = AUTSLVideoDataset(
                test_df,
                num_frames=self.config.num_frames,
                image_size=self.config.image_height,
                training=False
            )


            # ------------------------------------------------
            # Common DataLoader arguments
            # ------------------------------------------------

            loader_kwargs = {

                "batch_size":
                    self.config.batch_size,

                "num_workers":
                    self.config.num_workers,

                "pin_memory":
                    self.device.type == "cuda",

                "worker_init_fn":
                    self._seed_worker,
            }


            # ------------------------------------------------
            # Train loader
            # ------------------------------------------------

            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                **loader_kwargs
            )


            # ------------------------------------------------
            # Train evaluation loader
            # ------------------------------------------------

            train_eval_loader = DataLoader(
                train_eval_dataset,
                shuffle=False,
                **loader_kwargs
            )


            # ------------------------------------------------
            # Validation loader
            # ------------------------------------------------

            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                **loader_kwargs
            )


            # ------------------------------------------------
            # Test loader
            # ------------------------------------------------

            test_loader = DataLoader(
                test_dataset,
                shuffle=False,
                **loader_kwargs
            )


            logger.info(
                f"Train batches: "
                f"{len(train_loader)}"
            )

            logger.info(
                f"Train evaluation batches: "
                f"{len(train_eval_loader)}"
            )

            logger.info(
                f"Validation batches: "
                f"{len(val_loader)}"
            )

            logger.info(
                f"Test batches: "
                f"{len(test_loader)}"
            )


            return (
                train_loader,
                train_eval_loader,
                val_loader,
                test_loader
            )

        except Exception as e:

            logger.error(
                "Failed to create DataLoaders"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


    # ========================================================
    # Build R3D-18
    # ========================================================

    def _build_model(self):

        try:

            logger.info(
                "Loading R3D-18 Kinetics-400 "
                "pretrained model..."
            )

            weights = (
                R3D_18_Weights.KINETICS400_V1
            )

            model = r3d_18(
                weights=weights
            )


            # ------------------------------------------------
            # Replace classifier
            # ------------------------------------------------

            in_features = (
                model.fc.in_features
            )

            model.fc = nn.Sequential(

                nn.Dropout(
                    self.config.dropout_rate
                ),

                nn.Linear(
                    in_features,
                    self.config.num_classes
                )
            )


            # ------------------------------------------------
            # Freeze complete model
            # ------------------------------------------------

            for param in model.parameters():

                param.requires_grad = False


            # ------------------------------------------------
            # Unfreeze layer3
            # ------------------------------------------------

            for param in (
                model.layer3.parameters()
            ):

                param.requires_grad = True


            # ------------------------------------------------
            # Unfreeze layer4
            # ------------------------------------------------

            for param in (
                model.layer4.parameters()
            ):

                param.requires_grad = True


            # ------------------------------------------------
            # Unfreeze classifier
            # ------------------------------------------------

            for param in (
                model.fc.parameters()
            ):

                param.requires_grad = True


            total_params = sum(
                p.numel()
                for p in model.parameters()
            )

            trainable_params = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )


            logger.info(
                f"Total parameters: "
                f"{total_params:,}"
            )

            logger.info(
                f"Trainable parameters: "
                f"{trainable_params:,}"
            )

            logger.info(
                f"Trainable ratio: "
                f"{trainable_params / total_params:.4f}"
            )


            model = model.to(
                self.device
            )

            return model

        except Exception as e:

            logger.error(
                "Failed to build R3D-18 model"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


    # ========================================================
    # Compile Model
    # ========================================================

    def _compile_model(
        self,
        model
    ):

        try:

            criterion = nn.CrossEntropyLoss(
                label_smoothing=
                self.config.label_smoothing
            )


            optimizer = optim.AdamW(
                [
                    {
                        "params":
                            model.layer3.parameters(),

                        "lr":
                            self.config.layer3_learning_rate,
                    },

                    {
                        "params":
                            model.layer4.parameters(),

                        "lr":
                            self.config.layer4_learning_rate,
                    },

                    {
                        "params":
                            model.fc.parameters(),

                        "lr":
                            self.config.fc_learning_rate,
                    },
                ],

                weight_decay=
                    self.config.weight_decay
            )


            scheduler = (
                optim.lr_scheduler
                .CosineAnnealingWarmRestarts(
                    optimizer,

                    T_0=
                        self.config.scheduler_t0,

                    T_mult=
                        self.config.scheduler_t_mult,

                    eta_min=
                        self.config.scheduler_eta_min
                )
            )


            logger.info(
                "Optimizer and scheduler ready"
            )

            return (
                criterion,
                optimizer,
                scheduler
            )

        except Exception as e:

            logger.error(
                "Failed to compile model"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


    # ========================================================
    # Top-5 Metric
    # ========================================================

    @staticmethod
    def _compute_top5_correct(
        logits,
        labels
    ):

        k = min(
            5,
            logits.shape[1]
        )

        topk = (
            logits
            .topk(
                k,
                dim=1
            )
            .indices
        )

        correct = (
            topk
            .eq(
                labels.view(-1, 1)
            )
            .any(dim=1)
            .sum()
            .item()
        )

        return correct


    # ========================================================
    # Run One Epoch
    # ========================================================

    def _run_one_epoch(
        self,
        model,
        loader,
        criterion,
        optimizer=None,
        scheduler=None,
        epoch=0
    ):

        is_training = (
            optimizer is not None
        )


        if is_training:

            model.train()

        else:

            model.eval()


        total_loss = 0.0

        total_top5_correct = 0

        total_samples = 0

        all_predictions = []

        all_targets = []


        if is_training:

            optimizer.zero_grad(
                set_to_none=True
            )


        total_batches = len(
            loader
        )


        # ----------------------------------------------------
        # Batch loop
        # ----------------------------------------------------

        for step, batch in enumerate(
            loader
        ):

            videos, labels = batch


            videos = videos.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )


            with torch.set_grad_enabled(
                is_training
            ):

                with torch.amp.autocast(
                    device_type=self.amp_device,
                    enabled=self.use_amp
                ):

                    logits = model(
                        videos
                    )

                    loss = criterion(
                        logits,
                        labels
                    )


                if is_training:

                    scaled_loss = (
                        loss
                        / self.config.grad_accum_steps
                    )

                    self.scaler.scale(
                        scaled_loss
                    ).backward()


                    should_step = (

                        (
                            step + 1
                        )
                        %
                        self.config.grad_accum_steps
                        == 0

                        or

                        (
                            step + 1
                        )
                        == total_batches
                    )


                    if should_step:

                        self.scaler.unscale_(
                            optimizer
                        )


                        torch.nn.utils.clip_grad_norm_(
                            [
                                p
                                for p
                                in model.parameters()
                                if p.requires_grad
                            ],
                            self.config.gradient_clip_value
                        )


                        self.scaler.step(
                            optimizer
                        )

                        self.scaler.update()

                        optimizer.zero_grad(
                            set_to_none=True
                        )


            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            batch_size = (
                videos.size(0)
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_top5_correct += (
                self._compute_top5_correct(
                    logits.detach(),
                    labels
                )
            )

            total_samples += (
                batch_size
            )


            predictions = (
                logits
                .argmax(dim=1)
                .detach()
                .cpu()
                .numpy()
            )

            targets = (
                labels
                .detach()
                .cpu()
                .numpy()
            )


            all_predictions.extend(
                predictions
            )

            all_targets.extend(
                targets
            )


            # ------------------------------------------------
            # Live batch counter
            # ------------------------------------------------

            phase = (
                "Train"
                if is_training
                else "Validation"
            )

            print(
                f"\rEpoch {epoch + 1}/"
                f"{self.config.epochs} "
                f"[{phase}] | "
                f"Batch {step + 1}/"
                f"{total_batches}",
                end="",
                flush=True
            )


        print()


        # ----------------------------------------------------
        # Epoch metrics
        # ----------------------------------------------------

        average_loss = (
            total_loss
            / total_samples
        )

        accuracy = accuracy_score(
            all_targets,
            all_predictions
        )

        macro_f1 = f1_score(
            all_targets,
            all_predictions,
            average="macro",
            zero_division=0
        )

        top5_accuracy = (
            total_top5_correct
            / total_samples
        )


        return (
            average_loss,
            accuracy,
            macro_f1,
            top5_accuracy
        )


    # ========================================================
    # Save Best Checkpoint
    # ========================================================

    def _save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        best_val_f1,
        history
    ):

        try:

            checkpoint_dir = (
                self.config.checkpoint_dir
            )

            checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            checkpoint = {

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "best_val_f1":
                    best_val_f1,

                "config":
                    vars(self.config),

                "history":
                    history,
            }


            torch.save(
                checkpoint,
                self.config.checkpoint_file
            )


            logger.info(
                f"Best checkpoint saved: "
                f"{self.config.checkpoint_file}"
            )

        except Exception as e:

            logger.error(
                "Failed to save checkpoint"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


    # ========================================================
    # Training
    # ========================================================

    def _train_model(self):

        try:

            (
                train_loader,
                train_eval_loader,
                val_loader,
                test_loader
            ) = self._create_dataloaders()


            model = self._build_model()


            (
                criterion,
                optimizer,
                scheduler
            ) = self._compile_model(
                model
            )


            best_val_f1 = -1.0

            best_epoch = -1

            epochs_without_improvement = 0

            history = []


            # =================================================
            # Epoch Loop
            # =================================================

            for epoch in range(
                self.config.epochs
            ):

                start_time = time.time()


                print(
                    f"\nEpoch "
                    f"{epoch + 1}/"
                    f"{self.config.epochs}"
                )


                # ---------------------------------------------
                # Training
                # ---------------------------------------------

                (
                    train_loss,
                    train_acc,
                    train_f1,
                    train_top5
                ) = self._run_one_epoch(

                    model,

                    train_loader,

                    criterion,

                    optimizer=optimizer,

                    scheduler=scheduler,

                    epoch=epoch
                )


                # ---------------------------------------------
                # Validation
                # ---------------------------------------------

                (
                    val_loss,
                    val_acc,
                    val_f1,
                    val_top5
                ) = self._run_one_epoch(

                    model,

                    val_loader,

                    criterion,

                    optimizer=None,

                    scheduler=None,

                    epoch=epoch
                )


                # ---------------------------------------------
                # Scheduler
                #
                # Same epoch-level scheduler behavior as
                # notebook.
                # ---------------------------------------------

                scheduler.step(
                    epoch + 1
                )


                # ---------------------------------------------
                # Current learning rates
                # ---------------------------------------------

                current_lrs = [
                    group["lr"]
                    for group
                    in optimizer.param_groups
                ]


                epoch_time = (
                    time.time()
                    - start_time
                )


                epoch_result = {

                    "epoch":
                        epoch + 1,

                    "train_loss":
                        train_loss,

                    "train_accuracy":
                        train_acc,

                    "train_macro_f1":
                        train_f1,

                    "train_top5_accuracy":
                        train_top5,

                    "val_loss":
                        val_loss,

                    "val_accuracy":
                        val_acc,

                    "val_macro_f1":
                        val_f1,

                    "val_top5_accuracy":
                        val_top5,

                    "lr_layer3":
                        current_lrs[0],

                    "lr_layer4":
                        current_lrs[1],

                    "lr_fc":
                        current_lrs[2],

                    "epoch_time_seconds":
                        epoch_time,
                }


                history.append(
                    epoch_result
                )


                improved = (
                    val_f1
                    > best_val_f1
                )


                if improved:

                    best_val_f1 = (
                        val_f1
                    )

                    best_epoch = (
                        epoch + 1
                    )

                    epochs_without_improvement = 0

                    best_marker = " | best"


                    self._save_checkpoint(

                        model,

                        optimizer,

                        scheduler,

                        epoch + 1,

                        best_val_f1,

                        history
                    )

                else:

                    epochs_without_improvement += 1

                    best_marker = ""


                # ---------------------------------------------
                # Epoch Summary
                # ---------------------------------------------

                print(
                    f"Epoch "
                    f"{epoch + 1:02d} | "
                    f"Train Loss: "
                    f"{train_loss:.4f} | "
                    f"Train Acc: "
                    f"{train_acc:.4f} | "
                    f"Train F1: "
                    f"{train_f1:.4f} | "
                    f"Train Top5: "
                    f"{train_top5:.4f} | "
                    f"Val Loss: "
                    f"{val_loss:.4f} | "
                    f"Val Acc: "
                    f"{val_acc:.4f} | "
                    f"Val F1: "
                    f"{val_f1:.4f} | "
                    f"Val Top5: "
                    f"{val_top5:.4f} | "
                    f"Time: "
                    f"{epoch_time / 60:.2f} min"
                    f"{best_marker}"
                )


                # ---------------------------------------------
                # Early stopping
                # ---------------------------------------------

                if (
                    epochs_without_improvement
                    >=
                    self.config.early_stopping_patience
                ):

                    logger.info(
                        "Early stopping triggered."
                    )

                    break


            # =================================================
            # Training Complete
            # =================================================

            print(
                "\nTraining finished."
            )

            print(
                f"Best epoch: "
                f"{best_epoch}"
            )

            print(
                f"Best validation "
                f"macro-F1: "
                f"{best_val_f1:.6f}"
            )


            logger.info(
                f"Training completed. "
                f"Best epoch: {best_epoch}"
            )

            logger.info(
                f"Best validation Macro-F1: "
                f"{best_val_f1:.6f}"
            )


            # =================================================
            # Load Best Checkpoint
            # =================================================

            if not (
                self.config.checkpoint_file.exists()
            ):

                raise Exception(
                    "Best checkpoint was not created."
                )


            checkpoint = torch.load(
                self.config.checkpoint_file,
                map_location=self.device,
                weights_only=False
            )


            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )


            return (
                model,
                train_eval_loader,
                val_loader,
                test_loader,
                history
            )

        except Exception as e:

            logger.error(
                "Model training failed"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


    # ========================================================
    # Final Evaluation
    # ========================================================

    @torch.no_grad()
    def _evaluate(
        self,
        model,
        loader,
        criterion,
        split_name
    ):

        try:

            model.eval()


            total_loss = 0.0

            total_top5_correct = 0

            total_samples = 0

            all_predictions = []

            all_targets = []


            total_batches = len(
                loader
            )


            for batch_idx, (
                videos,
                labels
            ) in enumerate(
                loader
            ):

                videos = videos.to(
                    self.device,
                    non_blocking=True
                )

                labels = labels.to(
                    self.device,
                    non_blocking=True
                )


                with torch.amp.autocast(
                    device_type=self.amp_device,
                    enabled=self.use_amp
                ):

                    logits = model(
                        videos
                    )

                    loss = criterion(
                        logits,
                        labels
                    )


                batch_size = (
                    videos.size(0)
                )


                total_loss += (
                    loss.item()
                    * batch_size
                )


                total_top5_correct += (
                    self._compute_top5_correct(
                        logits,
                        labels
                    )
                )


                total_samples += (
                    batch_size
                )


                predictions = (
                    logits
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )

                targets = (
                    labels
                    .cpu()
                    .numpy()
                )


                all_predictions.extend(
                    predictions
                )

                all_targets.extend(
                    targets
                )


                print(
                    f"\rEvaluation [{split_name}] | "
                    f"Batch {batch_idx + 1}/"
                    f"{total_batches}",
                    end="",
                    flush=True
                )


            print()


            average_loss = (
                total_loss
                / total_samples
            )


            accuracy = accuracy_score(
                all_targets,
                all_predictions
            )


            macro_f1 = f1_score(
                all_targets,
                all_predictions,
                average="macro",
                zero_division=0
            )


            top5_accuracy = (
                total_top5_correct
                / total_samples
            )


            return (
                average_loss,
                accuracy,
                macro_f1,
                top5_accuracy
            )

        except Exception as e:

            logger.error(
                f"Evaluation failed for "
                f"{split_name}"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


    # ========================================================
    # Save Final Results
    # ========================================================

    def _save_final_results(
        self,
        results
    ):

        try:

            output_dir = (
                self.config.model_dir
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            results_path = (
                output_dir
                / "final_results.csv"
            )


            results_df = pd.DataFrame(
                results
            )


            results_df.to_csv(
                results_path,
                index=False
            )


            logger.info(
                f"Final results saved: "
                f"{results_path}"
            )


            return results_path

        except Exception as e:

            logger.error(
                "Failed to save final results"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


    # ========================================================
    # Save Model
    # ========================================================

    def _save_model(
        self,
        model
    ):

        try:

            model_dir = (
                self.config.model_dir
            )

            model_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            model_file = (
                self.config.model_file
            )


            torch.save(
                model.state_dict(),
                model_file
            )


            logger.info(
                f"Model saved successfully: "
                f"{model_file}"
            )


            return model_file

        except Exception as e:

            logger.error(
                "Failed to save final model"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )


    # ========================================================
    # Main Training Entry Point
    # ========================================================

    def initiate_model_training(self):

        try:

            (
                model,
                train_eval_loader,
                val_loader,
                test_loader,
                history
            ) = self._train_model()


            # ------------------------------------------------
            # Re-create criterion
            # ------------------------------------------------

            criterion = nn.CrossEntropyLoss(
                label_smoothing=
                self.config.label_smoothing
            )


            # ------------------------------------------------
            # Final deterministic train evaluation
            # ------------------------------------------------

            (
                train_loss,
                train_acc,
                train_f1,
                train_top5
            ) = self._evaluate(

                model,

                train_eval_loader,

                criterion,

                "Train"
            )


            # ------------------------------------------------
            # Final validation evaluation
            # ------------------------------------------------

            (
                val_loss,
                val_acc,
                val_f1,
                val_top5
            ) = self._evaluate(

                model,

                val_loader,

                criterion,

                "Validation"
            )


            # ------------------------------------------------
            # Final test evaluation
            # ------------------------------------------------

            (
                test_loss,
                test_acc,
                test_f1,
                test_top5
            ) = self._evaluate(

                model,

                test_loader,

                criterion,

                "Test"
            )


            # ------------------------------------------------
            # Results
            # ------------------------------------------------

            results = [

                {
                    "split":
                        "train",

                    "loss":
                        train_loss,

                    "accuracy":
                        train_acc,

                    "macro_f1":
                        train_f1,

                    "top5_accuracy":
                        train_top5,
                },

                {
                    "split":
                        "val",

                    "loss":
                        val_loss,

                    "accuracy":
                        val_acc,

                    "macro_f1":
                        val_f1,

                    "top5_accuracy":
                        val_top5,
                },

                {
                    "split":
                        "test",

                    "loss":
                        test_loss,

                    "accuracy":
                        test_acc,

                    "macro_f1":
                        test_f1,

                    "top5_accuracy":
                        test_top5,
                },
            ]


            results_path = (
                self._save_final_results(
                    results
                )
            )


            # ------------------------------------------------
            # Save final model
            # ------------------------------------------------

            model_file = (
                self._save_model(
                    model
                )
            )


            # ------------------------------------------------
            # Final console output
            # ------------------------------------------------

            print(
                "\nFinal Results"
            )

            print(
                pd.DataFrame(
                    results
                ).to_string(
                    index=False
                )
            )

            print(
                "\nNumbers to share with "
                "pose/SPOTER teammate:"
            )

            print(
                f"Train accuracy:   "
                f"{train_acc:.4f}"
            )

            print(
                f"Test accuracy:    "
                f"{test_acc:.4f}"
            )

            print(
                f"Test macro-F1:    "
                f"{test_f1:.4f}"
            )


            logger.info(
                "Model training and "
                "final evaluation completed successfully"
            )


            return model_file

        except Exception as e:

            logger.error(
                "Model training pipeline failed"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )