from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


NUM_CLASSES = 226
DROPOUT_RATE = 0.5

MODEL_PATH = Path(
    "artifacts/model/sign_language_model.pth"
)

OUTPUT_PATH = Path(
    "artifacts/model/sign_language_model.onnx"
)


def build_model():
    model = r3d_18(weights=None)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(
            in_features,
            NUM_CLASSES
        )
    )

    return model


def main():

    print("[INFO] Building R3D-18 model...")

    model = build_model()

    print("[INFO] Loading trained weights...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

    else:
        raise ValueError(
            "Unsupported model checkpoint format."
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    print("[INFO] PyTorch model loaded successfully.")

    # --------------------------------------------------------
    # Exact input shape used by your predictor
    # [Batch, Channels, Frames, Height, Width]
    # --------------------------------------------------------

    dummy_input = torch.randn(
        1,
        3,
        8,
        160,
        160,
        dtype=torch.float32
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("[INFO] Exporting model to ONNX...")

    with torch.no_grad():

        torch.onnx.export(
            model,
            dummy_input,
            OUTPUT_PATH,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["video"],
            output_names=["logits"],
            dynamic_axes={
                "video": {
                    0: "batch_size"
                },
                "logits": {
                    0: "batch_size"
                }
            }
        )

    print()
    print("=" * 60)
    print("ONNX EXPORT SUCCESSFUL")
    print("=" * 60)

    print(
        f"Output: {OUTPUT_PATH}"
    )

    size_mb = (
        OUTPUT_PATH.stat().st_size
        / (1024 * 1024)
    )

    print(
        f"Size: {size_mb:.2f} MB"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
