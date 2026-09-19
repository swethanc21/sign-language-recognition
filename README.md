# 🤟 Sign Language Recognition System

### AI-Powered Isolated Sign Language Recognition from Video

An AI-powered computer vision system that recognizes **isolated sign-language gestures from short video clips** using a deep learning-based **R3D-18 3D CNN** model.

The system analyzes a sequence of video frames to learn both **spatial features** and **temporal movement**, predicts the corresponding sign, and provides **confidence and Top-5 predictions** through an interactive web application.

---

## 🚀 Live Demo

🌐 **Web Application:**
https://sign-language-recognition-hp6elqxx8cs2ndyjtmmfzq.streamlit.app/

⚡ **FastAPI Backend:**
https://sign-language-recognition-71br.onrender.com

📚 **API Documentation:**
https://sign-language-recognition-71br.onrender.com/docs

---

## 🎯 Problem Statement

Communication can become difficult when a sign-language user interacts with someone who does not understand sign language.

Interpreters may not always be available, creating barriers in everyday communication, education, public services, and other situations.

This project explores an AI-based approach to recognize sign-language gestures from video and convert visual gesture information into an understandable digital prediction.

---

## 💡 Our Solution

The system takes a short video containing an isolated sign-language gesture and processes it through a deep learning pipeline.

### Workflow

```text
Video Input
     ↓
Frame Extraction
     ↓
8 Video Frames
     ↓
160 × 160 Preprocessing
     ↓
R3D-18 3D CNN
     ↓
Spatial + Temporal Feature Learning
     ↓
Sign Classification
     ↓
Predicted Sign
     ↓
Confidence + Top-5 Predictions
```

The 3D CNN allows the model to process information across multiple frames, helping it learn both the appearance of the gesture and its movement over time.

---

## 🧠 AI Model

| Component         | Details       |
| ----------------- | ------------- |
| Architecture      | R3D-18 3D CNN |
| Dataset           | AUTSL         |
| Number of Classes | 226           |
| Input             | 8 frames      |
| Frame Resolution  | 160 × 160     |
| Pretraining       | Kinetics-400  |
| Inference         | ONNX Runtime  |

---

## 📊 Model Performance

The current project model reports:

| Metric             |     Result |
| ------------------ | ---------: |
| **Test Accuracy**  | **34.90%** |
| **Macro F1**       | **32.44%** |
| **Top-5 Accuracy** | **65.53%** |

> These values represent the current model evaluation and are not intended to imply perfect recognition.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │       USER          │
                    │   Video / Gesture   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   STREAMLIT APP     │
                    │   User Interface    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FASTAPI         │
                    │      Backend        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    ONNX RUNTIME     │
                    │     Inference       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      R3D-18         │
                    │      3D CNN         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Sign + Confidence   │
                    │    + Top-5 Results  │
                    └─────────────────────┘
```

---

## ✨ Key Features

### 🎥 Video-Based Recognition

Recognizes isolated sign-language gestures from short video clips.

### 🧠 Spatial + Temporal Understanding

The R3D-18 architecture processes multiple frames together to learn visual and movement-related features.

### 🔢 226 Sign Classes

The current model supports recognition across 226 classes from the AUTSL dataset.

### 📈 Confidence & Top-5 Predictions

The application provides the predicted sign along with confidence information and alternative Top-5 predictions.

### 🌐 Interactive Web Application

A Streamlit interface allows users to interact with the recognition system.

### ⚡ API-Based Architecture

FastAPI provides a backend interface between the application and the model inference pipeline.

### 🚀 ONNX Inference

The deployed model uses ONNX Runtime for model inference.

---

## 🛠️ Technology Stack

### Programming

* Python

### AI / Deep Learning

* PyTorch
* Torchvision
* R3D-18 3D CNN
* AUTSL Dataset

### Computer Vision

* OpenCV

### Model Inference

* ONNX
* ONNX Runtime

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit

### Development

* Jupyter Notebook
* Git
* GitHub

---

## 📁 Project Structure

```text
sign-language-recognition/
│
├── api/
│   └── FastAPI backend
│
├── data/
│   └── Dataset / data-related files
│
├── notebook/
│   └── Jupyter notebooks
│
├── scripts/
│   └── Processing and utility scripts
│
├── src/
│   └── Sign-language recognition source code
│
├── streamlit_app/
│   └── Streamlit frontend
│
├── main.py
├── test.py
├── setup.py
├── requirements.txt
├── requirements-render.txt
├── packages.txt
├── .gitignore
└── README.md
```

---

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/swethanc21/sign-language-recognition.git
cd sign-language-recognition
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ⚡ Run the Backend

Start the FastAPI server:

```bash
uvicorn api.fastapi_app:app --host 127.0.0.1 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🖥️ Run the Streamlit Application

Open another terminal and activate the virtual environment.

Then run:

```bash
streamlit run streamlit_app/app.py
```

The Streamlit application will open in your browser.

---

## 🔮 Future Enhancements

The current project focuses on isolated sign-language recognition. Future development can explore:

* Improved model accuracy and robustness
* Larger sign-language vocabularies
* Continuous sign-language recognition
* Multilingual text output
* Text-to-speech integration
* Mobile and edge-device deployment
* Improved real-world lighting and background robustness
* Expanded datasets for broader accessibility

These are proposed future enhancements and are not part of the current implementation.

---

## 🌍 Potential Applications

The technology can potentially support:

* Accessibility-focused applications
* Educational tools
* Assistive communication interfaces
* Public-service accessibility
* Human-computer interaction
* Sign-language learning systems

---

## 🤝 Project Contributors

* **Swetha N.C.** — `@swethanc21`
* **Nalin N.C.** — `@Na-l-in-29`

---

## 📌 Project Information

**Project:** Sign Language Recognition System
**Event:** Hack Devengers 2.0
**Domain:** Artificial Intelligence / Computer Vision / Accessibility

---

## 📜 Disclaimer

This project is a technology prototype for isolated sign-language gesture recognition. It should not be considered a replacement for professional interpreters or a complete sign-language translation system.

---

## ⭐ Support the Project

If you find this project interesting, consider giving the repository a ⭐ on GitHub and exploring the live demo.

**GitHub:**
https://github.com/swethanc21/sign-language-recognition

**Live Demo:**
https://sign-language-recognition-hp6elqxx8cs2ndyjtmmfzq.streamlit.app/
