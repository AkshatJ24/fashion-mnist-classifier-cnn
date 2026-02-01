from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps
import io
import os
import sys

# 1. Setup Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
template_dir = os.path.join(project_root, 'templates')
static_dir = os.path.join(project_root, 'static')
model_path = os.path.join(project_root, 'fashion_model.onnx')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# 2. Load Model Safely
session = None
if os.path.exists(model_path):
    print(f"✅ Found model file at: {model_path}")
    try:
        session = ort.InferenceSession(model_path)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ CRITICAL ERROR: Model file NOT found at {model_path}")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image(image_bytes):
    # 1. Open and convert to Grayscale
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # 2. SMART INVERSION (The Fix)
    # Check if the image has a light background (mean pixel value > 127)
    # If yes, invert it to match Fashion MNIST (White item on Black bg)
    if np.mean(np.array(img)) > 127:
        img = ImageOps.invert(img)
    
    # 3. Resize to 28x28
    img = img.resize((28, 28))
    
    # 4. Normalize (0-1)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if session is None:
        return jsonify({'error': 'Model not loaded. Check server terminal logs.'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    try:
        input_data = preprocess_image(file.read())
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: input_data})
        
        # 1. Get the Index (0-9)
        predicted_idx = np.argmax(result[0])
        
        # 2. Calculate Confidence (Softmax)
        # We access result[0] which is shape (1, 10)
        exp_x = np.exp(result[0] - np.max(result[0]))
        probs = exp_x / np.sum(exp_x)
        
        # FIX IS HERE: We access the first row [0], then the specific class column [predicted_idx]
        confidence = float(probs[0][predicted_idx]) * 100
        
        return jsonify({
            'class': class_names[predicted_idx],
            'confidence': f"{confidence:.1f}%"
        })
    except Exception as e:
        # Print the full error to the terminal for debugging
        print(f"Error: {e}") 
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("\n🚀 Starting Server...")
    print("👉 Open this link: http://127.0.0.1:5001\n")
    app.run(debug=True, port=5001)