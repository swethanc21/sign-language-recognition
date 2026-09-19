from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from sign_language_detection.inference.predictor import (
    SignLanguagePredictor
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class TopPrediction(BaseModel):
    class_id: int
    sign: str
    confidence: float


class PredictionResponse(BaseModel):
    success: bool
    filename: str
    predicted_class: int
    predicted_sign: str
    confidence: float
    top_predictions: list[TopPrediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class RootResponse(BaseModel):
    message: str
    status: str
    model_status: str


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Sign Language Recognition API",
    description=(
        "REST API for isolated sign language recognition "
        "using a trained R3D-18 deep learning model."
    ),
    version="1.0.0"
)


# ============================================================
# MODEL INITIALIZATION
# ============================================================

try:

    predictor = SignLanguagePredictor(
        model_path=MODEL_PATH,
        class_mapping_path=CLASS_MAPPING_PATH
    )

    MODEL_STATUS = "loaded"

except Exception as error:

    predictor = None
    MODEL_STATUS = "failed"

    print(
        f"[ERROR] Failed to load model: {error}"
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get(
    "/",
    response_model=RootResponse
)
def root():

    return RootResponse(
        message="Sign Language Recognition API",
        status="running",
        model_status=MODEL_STATUS
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check():

    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None
    )


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict_sign(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if predictor is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    # --------------------------------------------------------
    # Check extension
    # --------------------------------------------------------

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension != ".mp4":

        raise HTTPException(
            status_code=400,
            detail="Only .mp4 video files are supported."
        )

    temporary_path = None

    try:

        # ----------------------------------------------------
        # Save uploaded video temporarily
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temporary_file:

            temporary_path = Path(
                temporary_file.name
            )

            shutil.copyfileobj(
                file.file,
                temporary_file
            )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        result = predictor.predict(
            temporary_path,
            top_k=5
        )

        # ----------------------------------------------------
        # Return structured response
        # ----------------------------------------------------

        return PredictionResponse(
            success=True,
            filename=file.filename,
            predicted_class=result[
                "predicted_class"
            ],
            predicted_sign=result[
                "predicted_sign"
            ],
            confidence=result[
                "confidence"
            ],
            top_predictions=[
                TopPrediction(
                    class_id=item["class_id"],
                    sign=item["sign"],
                    confidence=item["confidence"]
                )
                for item in result[
                    "top_predictions"
                ]
            ]
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )

    finally:

        # ----------------------------------------------------
        # Remove temporary video
        # ----------------------------------------------------

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):

            temporary_path.unlink()

        await file.close()


# ============================================================
# LOCAL DEVELOPMENT SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )