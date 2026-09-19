from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

from sign_language_detection.inference.predictor import (
    SignLanguagePredictor,
)


MODEL_PATH = Path(
    "artifacts/model/sign_language_model.pth"
)

MAPPING_PATH = Path(
    "data/mappings/SignList_ClassId_TR_EN.csv"
)

ONNX_PATH = Path(
    "artifacts/model/sign_language_model.onnx"
)


def main():

    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/compare_onnx.py <video.mp4>"
        )
        sys.exit(1)

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    print("[INFO] Loading PyTorch predictor...")

    predictor = SignLanguagePredictor(
        model_path=MODEL_PATH,
        class_mapping_path=MAPPING_PATH,
    )

    # --------------------------------------------------------
    # PyTorch prediction
    # --------------------------------------------------------

    print("\n[INFO] Running PyTorch inference...")

    pytorch_result = predictor.predict(
        video_path,
        top_k=5,
    )

    # --------------------------------------------------------
    # Create exactly the same input for ONNX
    # --------------------------------------------------------

    clip = predictor._load_video_clip(video_path)

    clip = clip.unsqueeze(0)

    onnx_input = (
        clip.cpu()
        .numpy()
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # ONNX Runtime
    # --------------------------------------------------------

    print("[INFO] Loading ONNX model...")

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    print("[INFO] Running ONNX inference...")

    outputs = session.run(
        ["logits"],
        {
            "video": onnx_input
        },
    )

    logits = outputs[0]

    probabilities = torch.softmax(
        torch.from_numpy(logits),
        dim=1,
    )

    top_probs, top_indices = torch.topk(
        probabilities,
        k=5,
        dim=1,
    )

    onnx_class = int(
        top_indices[0, 0].item()
    )

    onnx_sign = predictor.class_mapping[
        onnx_class
    ]

    onnx_confidence = float(
        top_probs[0, 0].item()
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    pytorch_class = pytorch_result[
        "predicted_class"
    ]

    pytorch_sign = pytorch_result[
        "predicted_sign"
    ]

    pytorch_confidence = pytorch_result[
        "confidence"
    ]

    print()
    print("=" * 60)
    print("PYTORCH vs ONNX")
    print("=" * 60)

    print(
        f"PyTorch : {pytorch_class} | "
        f"{pytorch_sign} | "
        f"{pytorch_confidence:.6f}"
    )

    print(
        f"ONNX    : {onnx_class} | "
        f"{onnx_sign} | "
        f"{onnx_confidence:.6f}"
    )

    print()

    if pytorch_class == onnx_class:
        print("[SUCCESS] Predictions MATCH")
    else:
        print("[WARNING] Predictions DO NOT MATCH")

    # --------------------------------------------------------
    # Numerical difference
    # --------------------------------------------------------

    pytorch_logits = None

    with torch.no_grad():
        torch_input = clip.to(
            predictor.device
        )

        pytorch_logits = (
            predictor.model(
                torch_input
            )
            .cpu()
            .numpy()
        )

    max_difference = np.max(
        np.abs(
            pytorch_logits - logits
        )
    )

    print(
        f"Max logit difference: "
        f"{max_difference:.8f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
