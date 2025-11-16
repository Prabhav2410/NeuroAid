#!/usr/bin/env python3
# xray_disease_prediction.py
"""
X-Ray Disease Prediction Module
Loads trained model and makes predictions on new X-ray images
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "xray_disease_model.h5"
CLASSES_PATH = "xray_disease_classes.json"
IMG_HEIGHT = 160
IMG_WIDTH = 160

# ============================================================
# LOAD MODEL AND CLASSES
# ============================================================

print("Loading X-ray model and classes...")

# Load model
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ X-ray model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading X-ray model: {e}")
    model = None

# Load class names
try:
    with open(CLASSES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"✅ Loaded {len(class_names)} X-ray classes")
except Exception as e:
    print(f"❌ Error loading X-ray classes: {e}")
    class_names = []

# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def validate_image(image_path):
    """
    Validate if image is suitable for X-ray prediction
    
    Args:
        image_path: Path to image file
        
    Returns:
        tuple: (is_valid, message)
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return False, "Image file not found"
        
        # Check file size (max 10MB)
        file_size = os.path.getsize(image_path)
        if file_size > 10 * 1024 * 1024:
            return False, "Image file too large (max 10MB)"
        
        # Try to open image
        img = Image.open(image_path)
        
        # Check if image can be loaded
        img.verify()
        
        # Reopen after verify (verify closes the file)
        img = Image.open(image_path)
        
        # Check image mode
        if img.mode not in ['RGB', 'L', 'RGBA']:
            return False, f"Unsupported image mode: {img.mode}"
        
        # Check image dimensions (should be reasonable)
        width, height = img.size
        if width < 50 or height < 50:
            return False, "Image too small (minimum 50x50 pixels)"
        
        if width > 5000 or height > 5000:
            return False, "Image too large (maximum 5000x5000 pixels)"
        
        return True, "Image is valid"
        
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def preprocess_image(image_path):
    """
    Preprocess image for X-ray model prediction
    
    Args:
        image_path: Path to image file
        
    Returns:
        np.array: Preprocessed image array ready for prediction
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        
        # Convert to array
        img_array = np.array(img)
        
        # Normalize pixel values (0-255 to 0-1)
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict(image_path):
    """
    Make prediction on X-ray image
    
    Args:
        image_path: Path to X-ray image
        
    Returns:
        list: List of (disease, confidence) tuples sorted by confidence
    """
    if model is None:
        raise Exception("Model not loaded")
    
    if not class_names:
        raise Exception("Class names not loaded")
    
    # Validate image
    is_valid, message = validate_image(image_path)
    if not is_valid:
        raise Exception(message)
    
    # Preprocess image
    img_array = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Get probabilities
    probabilities = predictions[0]
    
    # Create list of (disease, confidence) tuples
    results = []
    for i, prob in enumerate(probabilities):
        disease = class_names[i]
        confidence = float(prob) * 100  # Convert to percentage
        results.append((disease, confidence))
    
    # Sort by confidence (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def predict_with_details(image_path):
    """
    Make detailed prediction on X-ray image
    
    Args:
        image_path: Path to X-ray image
        
    Returns:
        dict: Dictionary with detailed prediction results
    """
    # Get predictions
    results = predict(image_path)
    
    # Get top prediction
    top_disease, top_confidence = results[0]
    
    # Determine if confident
    is_confident = top_confidence >= 60.0
    
    # Check if normal
    is_normal = top_disease.lower() == 'normal'
    
    # Determine severity
    if is_normal:
        severity = "Healthy - No Disease Detected"
    elif top_disease.lower() in ['covid-19', 'covid_19', 'covid19', 'covid']:
        severity = "Serious - Requires Immediate Attention"
    elif top_disease.lower() in ['tuberculosis', 'tb']:
        severity = "Serious - Requires Medical Treatment"
    elif top_disease.lower() == 'pneumonia':
        severity = "Moderate - Requires Medical Attention"
    else:
        severity = "Unknown Severity"
    
    # Format all predictions
    all_predictions = [
        {
            'disease': disease,
            'confidence': confidence
        }
        for disease, confidence in results
    ]
    
    return {
        'top_prediction': top_disease,
        'top_confidence': top_confidence,
        'is_confident': is_confident,
        'is_normal': is_normal,
        'severity': severity,
        'all_predictions': all_predictions
    }


# ============================================================
# TESTING
# ============================================================

def test_prediction():
    """Test the prediction function"""
    print("\n" + "=" * 60)
    print("X-RAY DISEASE PREDICTION TEST")
    print("=" * 60)
    
    if model is None:
        print("❌ Model not loaded. Please train the model first.")
        print("   Run: python xray_disease_trainer.py")
        return
    
    print(f"✅ Model loaded: {MODEL_PATH}")
    print(f"✅ Classes: {class_names}")
    print(f"✅ Number of classes: {len(class_names)}")
    
    # Check for test images
    test_dir = "xray_dataset/test"
    if not os.path.exists(test_dir):
        print(f"\n⚠️  Test directory not found: {test_dir}")
        print("   Please provide a test X-ray image path")
        return
    
    # Find a test image
    test_image = None
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                test_image = os.path.join(class_dir, images[0])
                break
    
    if test_image is None:
        print("\n⚠️  No test images found")
        return
    
    print(f"\n🔍 Testing with: {test_image}")
    
    try:
        # Make prediction
        results = predict_with_details(test_image)
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"\n🏆 Top Prediction: {results['top_prediction']}")
        print(f"📊 Confidence: {results['top_confidence']:.2f}%")
        print(f"✅ Is Confident: {results['is_confident']}")
        print(f"🩺 Is Normal: {results['is_normal']}")
        print(f"⚠️  Severity: {results['severity']}")
        
        print("\n📋 All Predictions:")
        print("-" * 60)
        for pred in results['all_predictions'][:5]:
            print(f"   {pred['disease']:20s} - {pred['confidence']:6.2f}%")
        
        print("\n✅ Prediction successful!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_prediction()