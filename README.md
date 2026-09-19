# 🤟 Sign Language Recognition

A deep learning-based **isolated sign language recognition system** that predicts sign-language gestures from short video clips using the **AUTSL dataset**.

## 🚀 Live Demo

**Web App:** https://sign-language-recognition-hp6elqxx8cs2ndyjtmmfzq.streamlit.app/

**API:** https://sign-language-recognition-71br.onrender.com

**Swagger Docs:** https://sign-language-recognition-71br.onrender.com/docs

## 🧠 Model

- **Architecture:** R3D-18 3D CNN
- **Pretrained on:** Kinetics-400
- **Dataset:** AUTSL
- **Classes:** 226
- **Input:** 8 frames, 160×160
- **Inference:** ONNX Runtime

## 📊 Performance

| Metric | Score |
|---|---:|
| Test Accuracy | **34.90%** |
| Macro F1 | **32.44%** |
| Top-5 Accuracy | **65.53%** |

## 🏗️ Architecture

```text
User
  ↓
Streamlit Cloud
  ↓
FastAPI (Render)
  ↓
ONNX Runtime
  ↓
R3D-18
  ↓
Sign + Confidence + Top-5
```

## 🛠️ Tech Stack

**Python · PyTorch · Torchvision · OpenCV · ONNX · ONNX Runtime · FastAPI · Streamlit**

## 📁 Project Structure

```text
sign-language-recognition/
├── api/
├── data/
├── scripts/
├── src/
├── streamlit_app/
├── requirements.txt
└── README.md
```

## ▶️ Run Locally

```bash
git clone https://github.com/ashu123243/sign-language-recognition.git
cd sign-language-recognition

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Start FastAPI:

```bash
uvicorn api.fastapi_app:app --host 127.0.0.1 --port 8000
```

Start Streamlit in another terminal:

```bash
streamlit run streamlit_app/app.py
```

## 🤗 Model

The deployed ONNX model is hosted on Hugging Face:

https://huggingface.co/pal-ashutosh-007/sign-language-recognition

## 👨‍💻 Author

**Ashutosh**

GitHub: https://github.com/ashu123243
