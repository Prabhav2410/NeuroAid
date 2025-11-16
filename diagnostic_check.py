#!/usr/bin/env python3
"""
NeuroAid Diagnostic Check - Updated with X-Ray Support
Run this to diagnose what's wrong with your setup
"""

import os
import sys

print("=" * 60)
print("NEUROAID DIAGNOSTIC CHECK")
print("=" * 60)

# Check 1: Required files
print("\n1. CHECKING REQUIRED FILES")
print("-" * 60)

required_files = {
    'Skin Disease Model': [
        'skin_disease_model.h5',
        'skin_disease_classes.json',
        'skin_disease_prediction.py',
        'skin_disease_trainer.py'
    ],
    'X-Ray Disease Model': [
        'xray_disease_model.h5',
        'xray_disease_classes.json',
        'xray_disease_prediction.py',
        'xray_disease_trainer.py'
    ],
    'Symptom Model': [
        'rf_disease_model.pkl',
        'symptom_list.pkl',
        'class_names.pkl',
        'disease_prediction.py',
        'symptoms.py'
    ],
    'Flask App': [
        'app.py',
        'templates/base.html',
        'templates/home.html',
        'templates/symptoms.html',
        'templates/skin_checker.html',
        'templates/xray_checker.html',
        'templates/xray_results.html',
        'templates/error.html'
    ],
    'Datasets': [
        'skin_disease_dataset/train',
        'skin_disease_dataset/test',
        'xray_dataset/train',
        'xray_dataset/test',
        'xray_dataset/val',
        'dataset_sorted.csv'
    ]
}

all_files_ok = True
for category, files in required_files.items():
    print(f"\n{category}:")
    for file in files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")
        if not exists:
            all_files_ok = False

# Check 2: Python imports
print("\n\n2. CHECKING PYTHON IMPORTS")
print("-" * 60)

imports_to_check = [
    ('flask', 'Flask'),
    ('tensorflow', 'TensorFlow'),
    ('PIL', 'Pillow (Image processing)'),
    ('numpy', 'NumPy'),
    ('sklearn', 'Scikit-learn'),
    ('joblib', 'Joblib'),
    ('pandas', 'Pandas'),
]

imports_ok = True
for module, name in imports_to_check:
    try:
        __import__(module)
        print(f"  ✅ {name}")
    except ImportError:
        print(f"  ❌ {name} - NOT INSTALLED")
        imports_ok = False

# Optional imports
print("\n  Optional packages:")
optional_imports = [
    ('google.generativeai', 'Google Generative AI (Chatbot)'),
]

for module, name in optional_imports:
    try:
        __import__(module)
        print(f"  ✅ {name}")
    except ImportError:
        print(f"  ⚠️  {name} - Not installed (optional)")

# Check 3: Skin Disease Model
print("\n\n3. CHECKING SKIN DISEASE MODEL")
print("-" * 60)

try:
    from skin_disease_prediction import predict_with_details, validate_image
    print("  ✅ skin_disease_prediction.py imports successfully")
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('skin_disease_model.h5')
        print("  ✅ Skin model loads successfully")
        print(f"  ✅ Model expects input shape: {model.input_shape}")
    except FileNotFoundError:
        print("  ❌ Skin model file not found: skin_disease_model.h5")
        print("     Run: python skin_disease_trainer.py")
    except Exception as e:
        print(f"  ⚠️ Skin model error: {e}")
        
except ImportError as e:
    print(f"  ❌ Cannot import skin_disease_prediction: {e}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Check 4: X-Ray Disease Model
print("\n\n4. CHECKING X-RAY DISEASE MODEL")
print("-" * 60)

try:
    from xray_disease_prediction import predict_with_details as xray_predict, validate_image as xray_validate
    print("  ✅ xray_disease_prediction.py imports successfully")
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('xray_disease_model.h5')
        print("  ✅ X-ray model loads successfully")
        print(f"  ✅ Model expects input shape: {model.input_shape}")
        
        # Load classes
        import json
        with open('xray_disease_classes.json', 'r') as f:
            classes = json.load(f)
        print(f"  ✅ X-ray classes: {classes}")
        
    except FileNotFoundError as e:
        print(f"  ❌ X-ray model file not found: {e}")
        print("     Run: python xray_disease_trainer.py")
    except Exception as e:
        print(f"  ⚠️ X-ray model error: {e}")
        
except ImportError as e:
    print(f"  ❌ Cannot import xray_disease_prediction: {e}")
    print("     Make sure xray_disease_prediction.py exists")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Check 5: Symptom Model
print("\n\n5. CHECKING SYMPTOM MODEL")
print("-" * 60)

try:
    from disease_prediction import predict_disease, validate_symptoms
    print("  ✅ disease_prediction.py imports successfully")
    
    try:
        import joblib
        model = joblib.load('rf_disease_model.pkl')
        symptoms = joblib.load('symptom_list.pkl')
        classes = joblib.load('class_names.pkl')
        print(f"  ✅ Symptom model loaded")
        print(f"  ✅ {len(symptoms)} symptoms available")
        print(f"  ✅ {len(classes)} diseases can be predicted")
        
    except FileNotFoundError as e:
        print(f"  ❌ Symptom model files not found: {e}")
        print("     Run: python symptoms.py")
    except Exception as e:
        print(f"  ⚠️ Symptom model error: {e}")
        
except ImportError as e:
    print(f"  ❌ Cannot import disease_prediction: {e}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Check 6: Upload folder
print("\n\n6. CHECKING UPLOAD FOLDER")
print("-" * 60)

if os.path.exists('uploads'):
    print("  ✅ uploads/ folder exists")
    # Check if writable
    try:
        test_file = os.path.join('uploads', '.test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("  ✅ uploads/ folder is writable")
    except Exception as e:
        print(f"  ⚠️ uploads/ folder not writable: {e}")
else:
    print("  ⚠️ uploads/ folder missing (will be created)")
    try:
        os.makedirs('uploads')
        print("  ✅ Created uploads/ folder")
    except Exception as e:
        print(f"  ❌ Cannot create uploads/ folder: {e}")

# Check 7: Dataset structure
print("\n\n7. CHECKING DATASET STRUCTURE")
print("-" * 60)

# Skin dataset
print("\nSkin Disease Dataset:")
skin_required = [
    'skin_disease_dataset/train',
    'skin_disease_dataset/test'
]
for path in skin_required:
    if os.path.exists(path):
        num_classes = len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        print(f"  ✅ {path} - {num_classes} classes")
    else:
        print(f"  ❌ {path} - Missing")

# X-ray dataset
print("\nX-Ray Dataset:")
xray_required = [
    'xray_dataset/train',
    'xray_dataset/test',
    'xray_dataset/val'
]
for path in xray_required:
    if os.path.exists(path):
        num_classes = len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        print(f"  ✅ {path} - {num_classes} classes")
    else:
        print(f"  ❌ {path} - Missing")
        
# Symptom dataset
if os.path.exists('dataset_sorted.csv'):
    import pandas as pd
    try:
        df = pd.read_csv('dataset_sorted.csv', nrows=5)
        print(f"\nSymptom Dataset:")
        print(f"  ✅ dataset_sorted.csv exists")
        print(f"  ✅ Columns: {list(df.columns)[:5]}...")
    except Exception as e:
        print(f"  ⚠️ Error reading dataset_sorted.csv: {e}")
else:
    print("\nSymptom Dataset:")
    print("  ❌ dataset_sorted.csv - Missing")

# Check 8: TensorFlow GPU
print("\n\n8. CHECKING GPU SUPPORT")
print("-" * 60)

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  ✅ Found {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"     GPU {i}: {gpu}")
        print("  ✅ GPU training available")
    else:
        print("  ⚠️ No GPU found - will use CPU")
        print("     For GPU: install CUDA and cuDNN")
except Exception as e:
    print(f"  ⚠️ Cannot check GPU: {e}")

# Check 9: Environment variables
print("\n\n9. CHECKING ENVIRONMENT VARIABLES")
print("-" * 60)

env_vars = {
    'GEMINI_API_KEY': 'Chatbot API key',
    'SECRET_KEY': 'Flask secret key'
}

for var, desc in env_vars.items():
    value = os.getenv(var)
    if value:
        masked = value[:10] + '...' if len(value) > 10 else value
        print(f"  ✅ {var} ({desc}): {masked}")
    else:
        print(f"  ⚠️ {var} ({desc}): Not set (optional)")

# Check 10: Templates
print("\n\n10. CHECKING TEMPLATES")
print("-" * 60)

templates = [
    'templates/base.html',
    'templates/home.html',
    'templates/symptoms.html',
    'templates/skin_checker.html',
    'templates/xray_checker.html',
    'templates/xray_results.html',
    'templates/error.html',
    'templates/login.html',
    'templates/about.html'
]

templates_ok = True
for template in templates:
    exists = os.path.exists(template)
    status = "✅" if exists else "❌"
    print(f"  {status} {template}")
    if not exists:
        templates_ok = False

# Final summary
print("\n\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)

issues = []

if not all_files_ok:
    issues.append("Missing required files")
    
if not imports_ok:
    issues.append("Missing Python packages")
    
if not templates_ok:
    issues.append("Missing template files")

if issues:
    print("❌ Issues found:")
    for issue in issues:
        print(f"   - {issue}")
    
    print("\n📋 NEXT STEPS:")
    print("-" * 60)
    
    if not imports_ok:
        print("\n1. Install missing packages:")
        print("   pip install flask tensorflow pillow numpy scikit-learn joblib pandas")
        print("   pip install google-generativeai  # Optional for chatbot")
    
    if not os.path.exists('skin_disease_model.h5'):
        print("\n2. Train skin disease model:")
        print("   python skin_disease_trainer.py")
    
    if not os.path.exists('xray_disease_model.h5'):
        print("\n3. Train X-ray disease model:")
        print("   python xray_disease_trainer.py")
    
    if not os.path.exists('rf_disease_model.pkl'):
        print("\n4. Train symptom model:")
        print("   python symptoms.py")
    
    if not templates_ok:
        print("\n5. Create missing templates")
        print("   Make sure all template files are in templates/ folder")
        
else:
    print("✅ All checks passed! Your setup looks good.")
    print("\n🚀 READY TO START:")
    print("-" * 60)
    print("\nTo start the application:")
    print("   python app.py")
    print("\nThen open your browser to:")
    print("   http://localhost:5000")
    print("\nAvailable features:")
    print("   ✓ Symptom Checker")
    print("   ✓ Skin Disease Checker")
    print("   ✓ X-Ray Disease Checker")
    if os.getenv('GEMINI_API_KEY'):
        print("   ✓ AI Chatbot")

print("\n" + "=" * 60 + "\n")

# Exit code
sys.exit(0 if not issues else 1)