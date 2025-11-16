# skin_disease_prediction.py
"""
Skin Disease Prediction from Images
Load trained model and make predictions on new images

Usage:
    python skin_disease_prediction.py <image_path>
    
Example:
    python skin_disease_prediction.py test_image.jpg
"""

import os
import numpy as np
import json
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "skin_disease_model.h5"
CLASS_NAMES_PATH = "skin_disease_classes.json"
IMG_HEIGHT = 160
IMG_WIDTH = 160

# ============================================================
# LOAD MODEL AND CLASSES
# ============================================================

print("Loading model and classes...")

# Load class names
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"❌ Class names file not found: {CLASS_NAMES_PATH}")

with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

print(f"✅ Loaded {len(class_names)} classes")

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}. Run skin_disease_trainer.py first.")

model = keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    """
    Load and preprocess image for prediction
    
    Args:
        image_path: Path to image file or PIL Image object
    
    Returns:
        Preprocessed image array
    """
    # Load image
    if isinstance(image_path, str):
        img = Image.open(image_path)
    else:
        img = image_path
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to array and normalize
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ============================================================
# PREDICTION
# ============================================================

def predict_skin_disease(image_path, top_k=3):
    """
    Predict skin disease from image
    
    Args:
        image_path: Path to image file or PIL Image object
        top_k: Number of top predictions to return
    
    Returns:
        List of tuples (disease_name, confidence_percentage)
    """
    try:
        # Preprocess image
        img_array = preprocess_image(image_path)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top K predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            disease = class_names[idx]
            confidence = float(predictions[idx]) * 100
            results.append((disease, confidence))
        
        return results
    
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return []

def predict_with_details(image_path, threshold=1.0):
    """
    Predict with detailed output including all classes above threshold
    
    Args:
        image_path: Path to image file or PIL Image object
        threshold: Minimum confidence percentage to include (default: 1.0%)
    
    Returns:
        Dictionary with detailed prediction info
    """
    try:
        # Preprocess image
        img_array = preprocess_image(image_path)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get all predictions above threshold
        results = []
        for idx, prob in enumerate(predictions):
            confidence = float(prob) * 100
            if confidence >= threshold:
                results.append({
                    'disease': class_names[idx],
                    'confidence': round(confidence, 2),
                    'confidence_raw': float(prob)
                })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get top prediction
        top_disease = results[0]['disease'] if results else "Unknown"
        top_confidence = results[0]['confidence'] if results else 0.0
        
        return {
            'top_prediction': top_disease,
            'top_confidence': top_confidence,
            'all_predictions': results,
            'is_confident': top_confidence >= 50.0  # Confidence flag
        }
    
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return {
            'top_prediction': "Error",
            'top_confidence': 0.0,
            'all_predictions': [],
            'is_confident': False,
            'error': str(e)
        }

# ============================================================
# BATCH PREDICTION
# ============================================================

def predict_batch(image_paths, top_k=3):
    """
    Predict multiple images at once
    
    Args:
        image_paths: List of image paths
        top_k: Number of top predictions per image
    
    Returns:
        List of prediction results
    """
    results = []
    for img_path in image_paths:
        try:
            predictions = predict_skin_disease(img_path, top_k)
            results.append({
                'image': img_path,
                'predictions': predictions,
                'success': True
            })
        except Exception as e:
            results.append({
                'image': img_path,
                'predictions': [],
                'success': False,
                'error': str(e)
            })
    
    return results

# ============================================================
# VALIDATION
# ============================================================

def validate_image(image_path):
    """
    Validate if image is suitable for prediction
    
    Args:
        image_path: Path to image file or PIL Image object
    
    Returns:
        Tuple (is_valid, error_message)
    """
    try:
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                return False, "Image file not found"
            
            # Check file size (max 10MB)
            file_size = os.path.getsize(image_path) / (1024 * 1024)
            if file_size > 10:
                return False, "Image file too large (max 10MB)"
            
            img = Image.open(image_path)
        else:
            img = image_path
        
        # Check if image can be converted to RGB
        if img.mode not in ['RGB', 'RGBA', 'L']:
            return False, "Unsupported image format"
        
        # Check dimensions
        width, height = img.size
        if width < 50 or height < 50:
            return False, "Image too small (minimum 50x50 pixels)"
        
        return True, "Valid image"
    
    except Exception as e:
        return False, f"Image validation error: {str(e)}"

# ============================================================
# CLI / EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("SKIN DISEASE PREDICTION")
    print("=" * 60 + "\n")
    
    if len(sys.argv) < 2:
        print("Usage: python skin_disease_prediction.py <image_path>")
        print("Example: python skin_disease_prediction.py test_image.jpg")
        print("\nOr test with dataset:")
        print("python skin_disease_prediction.py skin_disease_dataset/test/Acne/image1.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Validate image
    print(f"Validating image: {image_path}")
    is_valid, message = validate_image(image_path)
    if not is_valid:
        print(f"❌ {message}")
        sys.exit(1)
    print(f"✅ {message}\n")
    
    print(f"🔍 Analyzing image...\n")
    
    # Make prediction
    results = predict_with_details(image_path)
    
    print("=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\n🎯 Top Prediction: {results['top_prediction']}")
    print(f"📊 Confidence: {results['top_confidence']:.2f}%")
    print(f"✅ High Confidence: {'Yes' if results['is_confident'] else 'No'}")
    
    if len(results['all_predictions']) > 1:
        print(f"\n📋 Top 5 Predictions:")
        for i, pred in enumerate(results['all_predictions'][:5], 1):
            print(f"   {i}. {pred['disease']}: {pred['confidence']:.2f}%")
    
    print("\n" + "=" * 60)
    
    # Show warning if confidence is low
    if not results['is_confident']:
        print("\n⚠️  WARNING: Low confidence prediction!")
        print("   Consider consulting a healthcare professional for accurate diagnosis.")
        print("=" * 60)
    
    print("\n✅ Prediction complete!\n")