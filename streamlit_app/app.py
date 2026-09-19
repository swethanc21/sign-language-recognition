import subprocess
import tempfile
from pathlib import Path

import requests
import streamlit as st

import os

# ============================================================
# CONFIGURATION
# ============================================================

try:
    FASTAPI_URL = st.secrets["FASTAPI_URL"]
except Exception:
    FASTAPI_URL = os.getenv(
        "FASTAPI_URL",
        "http://127.0.0.1:8000/predict"
    )

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤟 Sign Language Recognition")

st.write(
    "Upload an isolated sign language video and let the "
    "trained deep learning model predict the most likely sign."
)


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a sign language video",
    type=["mp4"]
)


# ============================================================
# CREATE BROWSER-COMPATIBLE PREVIEW
# ============================================================

def create_video_preview(uploaded_video):

    input_path = None
    output_path = None

    try:

        # ----------------------------------------------------
        # Temporary input video
        # ----------------------------------------------------

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_path = Path(
            input_file.name
        )

        input_file.write(
            uploaded_video.getvalue()
        )

        input_file.close()

        # ----------------------------------------------------
        # Temporary output video
        # ----------------------------------------------------

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = Path(
            output_file.name
        )

        output_file.close()

        # ----------------------------------------------------
        # Convert to browser-compatible H.264 MP4
        # ----------------------------------------------------

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path)
        ]

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if process.returncode != 0:

            return None

        # ----------------------------------------------------
        # Read converted video
        # ----------------------------------------------------

        with open(
            output_path,
            "rb"
        ) as video_file:

            preview_bytes = (
                video_file.read()
            )

        return preview_bytes

    except Exception:

        return None

    finally:

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        if (
            input_path is not None
            and input_path.exists()
        ):

            input_path.unlink()

        if (
            output_path is not None
            and output_path.exists()
        ):

            output_path.unlink()


# ============================================================
# VIDEO PREVIEW
# ============================================================

if uploaded_file is not None:

    st.subheader("Video Preview")

    with st.spinner(
        "Preparing video preview..."
    ):

        preview_video = create_video_preview(
            uploaded_file
        )

    if preview_video is not None:

        st.video(
            preview_video,
            format="video/mp4"
        )

    else:

        st.warning(
            "The video could not be converted for browser "
            "preview. You can still try prediction."
        )

    st.caption(
        f"Uploaded Video: {uploaded_file.name}"
    )

    file_size_mb = (
        uploaded_file.size
        / (1024 * 1024)
    )

    st.caption(
        f"Video Size: {file_size_mb:.2f} MB"
    )


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    if st.button(
        "Predict Sign",
        type="primary",
        use_container_width=True
    ):

        try:

            # ------------------------------------------------
            # Reset file pointer
            # ------------------------------------------------

            uploaded_file.seek(0)

            # ------------------------------------------------
            # Prepare original video for FastAPI
            # ------------------------------------------------

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "video/mp4"
                )
            }

            # ------------------------------------------------
            # Send original video to FastAPI
            # ------------------------------------------------

            with st.spinner(
                "Analyzing video..."
            ):

                response = requests.post(
                    FASTAPI_URL,
                    files=files,
                    timeout=300
                )

            # ------------------------------------------------
            # Successful prediction
            # ------------------------------------------------

            if response.status_code == 200:

                result = response.json()

                predicted_sign = result[
                    "predicted_sign"
                ]

                confidence = result[
                    "confidence"
                ]

                top_predictions = result[
                    "top_predictions"
                ]

                # ------------------------------------------------
                # Prediction Result
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "Prediction Result"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Predicted Sign",
                        predicted_sign.upper()
                    )

                with col2:

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                # ------------------------------------------------
                # Confidence Progress
                # ------------------------------------------------

                st.progress(
                    float(confidence)
                )

                # ------------------------------------------------
                # Confidence Information
                # ------------------------------------------------

                if confidence < 0.10:

                    st.warning(
                        "The model is uncertain about this "
                        "prediction. Check the alternative "
                        "predictions below."
                    )

                elif confidence < 0.25:

                    st.info(
                        "The model shows moderate confidence. "
                        "The top alternatives may also be relevant."
                    )

                else:

                    st.success(
                        "The model produced a relatively "
                        "strong prediction."
                    )

                # ------------------------------------------------
                # Top 5 Predictions
                # ------------------------------------------------

                st.subheader(
                    "Top 5 Predictions"
                )

                for rank, prediction in enumerate(
                    top_predictions,
                    start=1
                ):

                    sign = prediction[
                        "sign"
                    ]

                    probability = prediction[
                        "confidence"
                    ]

                    st.write(
                        f"**{rank}. {sign.upper()}** — "
                        f"{probability * 100:.2f}%"
                    )

                    st.progress(
                        float(probability)
                    )

            # ------------------------------------------------
            # API Error
            # ------------------------------------------------

            else:

                st.error(
                    "Prediction failed."
                )

                try:

                    error_message = (
                        response.json()
                        .get(
                            "detail",
                            "Unknown API error."
                        )
                    )

                    st.write(
                        error_message
                    )

                except Exception:

                    st.write(
                        response.text
                    )

        # ----------------------------------------------------
        # FastAPI connection error
        # ----------------------------------------------------

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the prediction server."
            )

            st.info(
                "Please make sure FastAPI is running on "
                "http://127.0.0.1:8000"
            )

        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        except requests.exceptions.Timeout:

            st.error(
                "The prediction request timed out."
            )

        # ----------------------------------------------------
        # Unexpected error
        # ----------------------------------------------------

        except Exception as error:

            st.error(
                f"Unexpected error: {error}"
            )


# ============================================================
# CLEAR
# ============================================================

if st.button(
    "Clear"
):

    st.rerun()


# ============================================================
# ABOUT THE MODEL
# ============================================================

st.divider()

st.subheader(
    "About the Model"
)

st.write(
    "This application performs isolated sign language "
    "recognition on the AUTSL dataset using a video "
    "classification model."
)


# ============================================================
# MODEL METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Classes",
        "226"
    )

with col2:

    st.metric(
        "Test Accuracy",
        "34.90%"
    )

with col3:

    st.metric(
        "Macro F1-Score",
        "32.44%"
    )


# ============================================================
# MODEL DETAILS
# ============================================================

st.write(
    "**Architecture:** R3D-18 with Kinetics-400 pretraining"
)

st.write(
    "**Input:** 8 frames per video at 160 × 160 resolution"
)

st.write(
    "**Evaluation:** Signer-independent AUTSL test split"
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Sign Language Recognition • AUTSL • FastAPI • Streamlit"
)