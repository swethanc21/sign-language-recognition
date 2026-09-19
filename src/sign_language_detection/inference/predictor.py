from pathlib import Path
from urllib.request import urlretrieve
import csv

import cv2
import numpy as np
import onnxruntime as ort

HUGGINGFACE_MODEL_URL = (
    "https://huggingface.co/pal-ashutosh-007/"
    "sign-language-recognition/resolve/main/"
    "sign_language_model.onnx"
)

HUGGINGFACE_MODEL_DATA_URL = (
    "https://huggingface.co/pal-ashutosh-007/"
    "sign-language-recognition/resolve/main/"
    "sign_language_model.onnx.data"
)
# ============================================================
# KINETICS-400 NORMALIZATION
# Same normalization used during model training
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
# HUGGING FACE MODEL
# ============================================================

HUGGINGFACE_MODEL_URL = (
    "https://huggingface.co/"
    "pal-ashutosh-007/"
    "sign-language-recognition/"
    "resolve/main/"
    "sign_language_model.onnx"
)


class SignLanguagePredictor:
    """
    ONNX Runtime inference service for the trained
    AUTSL Sign Language Recognition model.
    """

    def __init__(
        self,
        model_path: str | Path,
        class_mapping_path: str | Path,
        num_classes: int = 226,
        num_frames: int = 8,
        image_size: int = 160,
    ):
        self.model_path = Path(model_path)
        self.class_mapping_path = Path(class_mapping_path)

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.image_size = image_size

        # ----------------------------------------------------
        # Load class mapping
        # ----------------------------------------------------

        self.class_mapping = self._load_class_mapping()

        # ----------------------------------------------------
        # Load ONNX model
        # ----------------------------------------------------

        self.session = self._load_model()

        # ----------------------------------------------------
        # Input / output names
        # ----------------------------------------------------

        self.input_name = (
            self.session.get_inputs()[0].name
        )

        self.output_name = (
            self.session.get_outputs()[0].name
        )

        print(
            "[INFO] ONNX Runtime inference provider: "
            f"{self.session.get_providers()}"
        )

        print(
            "[INFO] Sign Language ONNX model loaded successfully."
        )

        print(
            f"[INFO] Class mapping loaded: "
            f"{len(self.class_mapping)} classes"
        )

    # ========================================================
    # CLASS MAPPING
    # ========================================================

    def _load_class_mapping(self):
        """
        Load ClassId -> English sign name mapping
        using Python's built-in csv module.
        """

        if not self.class_mapping_path.exists():
            raise FileNotFoundError(
                "Class mapping file not found: "
                f"{self.class_mapping_path}"
            )

        mapping = {}

        with open(
            self.class_mapping_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ValueError(
                    "Class mapping CSV has no header."
                )

            required_columns = {
                "ClassId",
                "EN"
            }

            if not required_columns.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    "Class mapping CSV must contain "
                    "'ClassId' and 'EN' columns."
                )

            for row in reader:

                try:
                    class_id = int(
                        row["ClassId"]
                    )

                except (
                    TypeError,
                    ValueError
                ) as error:

                    raise ValueError(
                        "Invalid ClassId in mapping CSV."
                    ) from error

                sign = (
                    str(row["EN"])
                    .strip()
                )

                mapping[class_id] = sign

        # ----------------------------------------------------
        # Validate number of classes
        # ----------------------------------------------------

        if len(mapping) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes, "
                f"but found {len(mapping)}."
            )

        # ----------------------------------------------------
        # Validate class IDs
        # ----------------------------------------------------

        expected_ids = set(
            range(self.num_classes)
        )

        actual_ids = set(
            mapping.keys()
        )

        if actual_ids != expected_ids:
            raise ValueError(
                "Class mapping IDs do not match "
                f"0-{self.num_classes - 1}."
            )

        return mapping

    # ========================================================
    # DOWNLOAD MODEL
    # ========================================================

    def _download_model(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data_path = self.model_path.with_name(
            self.model_path.name + ".data"
        )

        if not self.model_path.exists():
            print("[INFO] Downloading ONNX model...")
            urlretrieve(
                HUGGINGFACE_MODEL_URL,
                self.model_path
            )
            print("[INFO] ONNX model downloaded.")

        if not model_data_path.exists():
            print("[INFO] Downloading ONNX external data...")
            urlretrieve(
                HUGGINGFACE_MODEL_DATA_URL,
                model_data_path
            )
            print("[INFO] ONNX external data downloaded.")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def _load_model(self):
        """
        Load the ONNX model using ONNX Runtime.

        If the model is not available locally,
        automatically download it from Hugging Face.
        """

        if self.model_path.exists():

            print(
                "[INFO] Local ONNX model file found."
            )

        else:

            self._download_model()

        try:

            session = ort.InferenceSession(
                str(self.model_path),
                providers=[
                    "CPUExecutionProvider"
                ]
            )

        except Exception as error:

            raise RuntimeError(
                "Failed to load ONNX model: "
                f"{error}"
            ) from error

        return session

    # ========================================================
    # FRAME SAMPLING
    # ========================================================

    def _sample_frame_indices(
        self,
        total_frames: int
    ):

        if total_frames <= 0:
            return [0] * self.num_frames

        boundaries = np.linspace(
            0,
            total_frames,
            self.num_frames + 1
        ).astype(int)

        indices = []

        for i in range(
            self.num_frames
        ):

            start = boundaries[i]
            end = boundaries[i + 1]

            if end > start:

                index = (
                    start + end - 1
                ) // 2

            else:

                index = min(
                    start,
                    total_frames - 1
                )

            index = max(
                0,
                min(
                    index,
                    total_frames - 1
                )
            )

            indices.append(
                index
            )

        return indices

    # ========================================================
    # BLACK CLIP
    # ========================================================

    def _make_black_clip(self):

        black = np.zeros(
            (
                self.num_frames,
                self.image_size,
                self.image_size,
                3
            ),
            dtype=np.float32
        )

        black = (
            black - KINETICS_MEAN
        ) / KINETICS_STD

        return black.transpose(
            3, 0, 1, 2
        ).astype(np.float32)

    # ========================================================
    # VIDEO LOADING
    # ========================================================

    def _load_video_clip(
        self,
        video_path: str | Path
    ):

        video_path = str(
            video_path
        )

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            print(
                "[WARNING] Could not open video: "
                f"{video_path}"
            )

            return self._make_black_clip()

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total_frames <= 0:

            print(
                "[WARNING] Invalid frame count: "
                f"{video_path}"
            )

            cap.release()

            return self._make_black_clip()

        frame_indices = (
            self._sample_frame_indices(
                total_frames
            )
        )

        frames = []
        last_valid_frame = None

        try:

            for index in frame_indices:

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    index
                )

                success, frame = (
                    cap.read()
                )

                if not success:

                    if (
                        last_valid_frame
                        is not None
                    ):

                        frame = (
                            last_valid_frame.copy()
                        )

                    else:

                        frame = np.zeros(
                            (
                                self.image_size,
                                self.image_size,
                                3
                            ),
                            dtype=np.uint8
                        )

                else:

                    last_valid_frame = (
                        frame.copy()
                    )

                # BGR -> RGB
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Resize
                frame = cv2.resize(
                    frame,
                    (
                        self.image_size,
                        self.image_size
                    ),
                    interpolation=cv2.INTER_LINEAR
                )

                # [0,255] -> [0,1]
                frame = (
                    frame.astype(
                        np.float32
                    ) / 255.0
                )

                frames.append(
                    frame
                )

        except Exception as error:

            print(
                "[WARNING] Video processing failed: "
                f"{video_path}"
            )

            print(
                f"[WARNING] Error: {error}"
            )

            cap.release()

            return self._make_black_clip()

        finally:

            cap.release()

        if len(frames) != self.num_frames:

            while (
                len(frames)
                < self.num_frames
            ):

                if frames:

                    frames.append(
                        frames[-1].copy()
                    )

                else:

                    frames.append(
                        np.zeros(
                            (
                                self.image_size,
                                self.image_size,
                                3
                            ),
                            dtype=np.float32
                        )
                    )

            frames = frames[
                :self.num_frames
            ]

        # [T,H,W,C]
        clip = np.stack(
            frames,
            axis=0
        )

        # Kinetics normalization
        clip = (
            clip - KINETICS_MEAN
        ) / KINETICS_STD

        # [T,H,W,C] -> [C,T,H,W]
        clip = clip.transpose(
            3, 0, 1, 2
        ).astype(
            np.float32
        )

        return clip

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        video_path: str | Path,
        top_k: int = 5
    ):
        """
        Predict sign class from a video.
        """

        video_path = Path(
            video_path
        )

        if not video_path.exists():

            raise FileNotFoundError(
                f"Video file not found: "
                f"{video_path}"
            )

        # ----------------------------------------------------
        # Load clip
        # ----------------------------------------------------

        clip = self._load_video_clip(
            video_path
        )

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        clip = np.expand_dims(
            clip,
            axis=0
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # ONNX inference
        # ----------------------------------------------------

        logits = self.session.run(
            [self.output_name],
            {
                self.input_name: clip
            }
        )[0]

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        logits = logits[0]

        logits = (
            logits
            - np.max(logits)
        )

        exp_logits = np.exp(
            logits
        )

        probabilities = (
            exp_logits
            / np.sum(exp_logits)
        )

        # ----------------------------------------------------
        # Top-K
        # ----------------------------------------------------

        k = min(
            top_k,
            self.num_classes
        )

        top_indices = np.argsort(
            probabilities
        )[::-1][:k]

        # ----------------------------------------------------
        # Main prediction
        # ----------------------------------------------------

        predicted_class = int(
            top_indices[0]
        )

        predicted_sign = (
            self.class_mapping[
                predicted_class
            ]
        )

        confidence = float(
            probabilities[
                predicted_class
            ]
        )

        # ----------------------------------------------------
        # Top predictions
        # ----------------------------------------------------

        top_predictions = []

        for class_id in top_indices:

            class_id = int(
                class_id
            )

            top_predictions.append(
                {
                    "class_id": class_id,
                    "sign": self.class_mapping[
                        class_id
                    ],
                    "confidence": float(
                        probabilities[
                            class_id
                        ]
                    )
                }
            )

        return {
            "predicted_class": predicted_class,
            "predicted_sign": predicted_sign,
            "confidence": confidence,
            "top_predictions": top_predictions
        }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    PROJECT_ROOT = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    MODEL_PATH = (
        PROJECT_ROOT
        / "artifacts"
        / "model"
        / "sign_language_model.onnx"
    )

    CLASS_MAPPING_PATH = (
        PROJECT_ROOT
        / "data"
        / "mappings"
        / "SignList_ClassId_TR_EN.csv"
    )

    predictor = SignLanguagePredictor(
        model_path=MODEL_PATH,
        class_mapping_path=CLASS_MAPPING_PATH
    )

    print()
    print("=" * 60)
    print("ONNX MODEL + CLASS MAPPING TEST")
    print("=" * 60)

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Mapping: {CLASS_MAPPING_PATH}"
    )

    print(
        f"Classes: {len(predictor.class_mapping)}"
    )

    print(
        f"Providers: {predictor.session.get_providers()}"
    )

    print("Status: SUCCESS")

    print("=" * 60)