# 👕 Fashion MNIST Classifier (End-to-End AI Web App)

A full-stack AI application that classifies clothing images in real-time. Built with **TensorFlow**, optimized with **ONNX**, and deployed using **Flask**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-purple)
![Flask](https://img.shields.io/badge/Backend-Flask-green)

## 🚀 Live Demo
[Link to your Vercel App will go here]

## 💡 Key Features
* **Custom CNN Architecture:** Trained on the Fashion MNIST dataset (60,000 images) using Conv2D, MaxPooling, and Dropout layers.
* **ONNX Optimization:** Converted the raw TensorFlow model to `.onnx` format, reducing inference latency and stripping training-only layers (RandomFlip/Rotation).
* **Smart Preprocessing:** Implements **Domain Adaptation** to handle real-world photos. The app automatically detects white backgrounds (common in product photos) and inverts them to match the black-background training data.
* **Responsive UI:** A clean, modern frontend that allows users to upload images and view confidence scores instantly.

## 🛠️ Tech Stack
* **Training:** Python, TensorFlow, Keras
* **Inference:** ONNX Runtime (CPU optimized)
* **Backend:** Flask (Python)
* **Frontend:** HTML5, CSS3, JavaScript

## 📂 Project Structure
```text
/fashion-project
  ├── fashion_model.onnx    # Optimized Model (Product of training)
  ├── train.py              # Training pipeline (CNN -> Export -> ONNX)
  ├── requirements.txt      # Dependencies
  ├── api/
  │   └── index.py          # Flask Backend & Inference Logic
  └── templates/
      └── index.html        # Frontend UI
