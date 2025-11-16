#!/usr/bin/env python3
# verify_xray_dataset.py
"""
X-Ray Dataset Verification Tool
Checks for:
- Class distribution imbalance
- Duplicate images
- Image quality issues
- Proper train/val/test split
"""

import os
import hashlib
from PIL import Image
from collections import Counter, defaultdict
import json

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "xray_dataset/train"
VAL_DIR = "xray_dataset/val"
TEST_DIR = "xray_dataset/test"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_hash(image_path):
    """Calculate hash of image for duplicate detection"""
    try:
        with Image.open(image_path) as img:
            # Resize to small size for faster hashing
            img_small = img.resize((8, 8), Image.Resampling.LANCZOS)
            # Convert to grayscale
            img_gray = img_small.convert('L')
            # Get hash
            img_hash = hashlib.md5(img_gray.tobytes()).hexdigest()
            return img_hash
    except Exception as e:
        print(f"   ⚠️  Error hashing {image_path}: {e}")
        return None

def check_image_quality(image_path):
    """Check if image is valid and get its properties"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            
            issues = []
            
            # Check if too small
            if width < 50 or height < 50:
                issues.append("TOO_SMALL")
            
            # Check if unusual size
            if width > 5000 or height > 5000:
                issues.append("TOO_LARGE")
            
            # Check aspect ratio
            aspect_ratio = width / height
            if aspect_ratio > 3 or aspect_ratio < 0.33:
                issues.append("UNUSUAL_ASPECT_RATIO")
            
            return {
                'valid': True,
                'width': width,
                'height': height,
                'mode': mode,
                'issues': issues
            }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'issues': ['CORRUPT']
        }

# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyze_directory(data_dir, dir_name):
    """Analyze a dataset directory"""
    print("\n" + "=" * 60)
    print(f"ANALYZING {dir_name.upper()} DIRECTORY")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return None
    
    class_info = {}
    all_hashes = []
    total_images = 0
    corrupt_images = 0
    
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not classes:
        print(f"❌ No class directories found in {data_dir}")
        return None
    
    print(f"\nFound {len(classes)} classes: {classes}\n")
    
    for class_name in sorted(classes):
        class_path = os.path.join(data_dir, class_name)
        
        print(f"Analyzing class: {class_name}")
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        class_hashes = []
        class_issues = []
        
        for img_file in images:
            img_path = os.path.join(class_path, img_file)
            
            # Check image quality
            quality = check_image_quality(img_path)
            
            if not quality['valid']:
                corrupt_images += 1
                class_issues.append({
                    'file': img_file,
                    'issues': ['CORRUPT'],
                    'error': quality.get('error', 'Unknown error')
                })
                continue
            
            if quality['issues']:
                class_issues.append({
                    'file': img_file,
                    'issues': quality['issues']
                })
            
            # Calculate hash for duplicate detection
            img_hash = get_image_hash(img_path)
            if img_hash:
                class_hashes.append(img_hash)
                all_hashes.append((img_hash, class_name, img_file))
        
        class_info[class_name] = {
            'count': len(images),
            'valid_count': len(class_hashes),
            'hashes': class_hashes,
            'issues': class_issues
        }
        
        total_images += len(images)
        
        print(f"   ✅ {len(images)} images found")
        if class_issues:
            print(f"   ⚠️  {len(class_issues)} images with issues")
    
    print(f"\n📊 Total images: {total_images}")
    if corrupt_images > 0:
        print(f"⚠️  Corrupt images: {corrupt_images}")
    
    return {
        'total': total_images,
        'classes': class_info,
        'hashes': all_hashes,
        'corrupt': corrupt_images
    }

def check_class_balance(train_info, val_info, test_info):
    """Check for class imbalance across splits"""
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    all_classes = set()
    if train_info:
        all_classes.update(train_info['classes'].keys())
    if val_info:
        all_classes.update(val_info['classes'].keys())
    if test_info:
        all_classes.update(test_info['classes'].keys())
    
    print("\nClass Distribution Across Splits:")
    print("-" * 60)
    print(f"{'Class':<20} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print("-" * 60)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for class_name in sorted(all_classes):
        train_count = train_info['classes'][class_name]['count'] if train_info and class_name in train_info['classes'] else 0
        val_count = val_info['classes'][class_name]['count'] if val_info and class_name in val_info['classes'] else 0
        test_count = test_info['classes'][class_name]['count'] if test_info and class_name in test_info['classes'] else 0
        
        total = train_count + val_count + test_count
        
        print(f"{class_name:<20} {train_count:>10} {val_count:>10} {test_count:>10} {total:>10}")
        
        total_train += train_count
        total_val += val_count
        total_test += test_count
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_train:>10} {total_val:>10} {total_test:>10} {total_train + total_val + total_test:>10}")
    
    # Calculate imbalance
    print("\n" + "=" * 60)
    print("IMBALANCE ANALYSIS (Training Set)")
    print("=" * 60)
    
    if train_info:
        counts = [info['count'] for info in train_info['classes'].values()]
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nMax samples per class: {max_count}")
        print(f"Min samples per class: {min_count}")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}x")
        
        if imbalance_ratio > 3.0:
            print("\n⚠️  SEVERE IMBALANCE! This will cause biased predictions!")
            print("   Recommendation: Use class weights or resample data")
        elif imbalance_ratio > 2.0:
            print("\n⚠️  Moderate imbalance detected")
            print("   Recommendation: Use class weights during training")
        else:
            print("\n✅ Dataset is reasonably balanced")
    
    print("=" * 60)

def check_duplicates(train_info, val_info, test_info):
    """Check for duplicate images within and across splits"""
    print("\n" + "=" * 60)
    print("DUPLICATE DETECTION")
    print("=" * 60)
    
    all_hashes = []
    
    if train_info:
        for hash_val, class_name, file_name in train_info['hashes']:
            all_hashes.append((hash_val, 'train', class_name, file_name))
    
    if val_info:
        for hash_val, class_name, file_name in val_info['hashes']:
            all_hashes.append((hash_val, 'val', class_name, file_name))
    
    if test_info:
        for hash_val, class_name, file_name in test_info['hashes']:
            all_hashes.append((hash_val, 'test', class_name, file_name))
    
    # Find duplicates
    hash_dict = defaultdict(list)
    for hash_val, split, class_name, file_name in all_hashes:
        hash_dict[hash_val].append((split, class_name, file_name))
    
    duplicates = {k: v for k, v in hash_dict.items() if len(v) > 1}
    
    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} duplicate image groups!\n")
        
        # Check for cross-split duplicates (data leakage)
        cross_split_duplicates = []
        within_split_duplicates = []
        
        for hash_val, instances in duplicates.items():
            splits = set(inst[0] for inst in instances)
            if len(splits) > 1:
                cross_split_duplicates.append((hash_val, instances))
            else:
                within_split_duplicates.append((hash_val, instances))
        
        if cross_split_duplicates:
            print(f"🚨 CRITICAL: {len(cross_split_duplicates)} duplicate groups across train/val/test!")
            print("   This causes DATA LEAKAGE and inflates accuracy!\n")
            
            print("Examples of cross-split duplicates:")
            for i, (hash_val, instances) in enumerate(cross_split_duplicates[:5]):
                print(f"\n   Duplicate group {i+1}:")
                for split, class_name, file_name in instances:
                    print(f"      - {split}/{class_name}/{file_name}")
            
            if len(cross_split_duplicates) > 5:
                print(f"\n   ... and {len(cross_split_duplicates) - 5} more groups")
        
        if within_split_duplicates:
            print(f"\nℹ️  {len(within_split_duplicates)} duplicate groups within same split")
            print("   (Less critical, but reduces effective training data)")
    else:
        print("\n✅ No duplicates found!")
    
    print("=" * 60)

def generate_report(train_info, val_info, test_info):
    """Generate summary report"""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION REPORT")
    print("=" * 60)
    
    issues = []
    warnings = []
    
    # Check if all splits exist
    if not train_info:
        issues.append("❌ Training set not found or empty")
    if not val_info:
        warnings.append("⚠️  Validation set not found or empty")
    if not test_info:
        warnings.append("⚠️  Test set not found or empty")
    
    # Check class consistency
    if train_info and val_info:
        train_classes = set(train_info['classes'].keys())
        val_classes = set(val_info['classes'].keys())
        
        if train_classes != val_classes:
            issues.append(f"❌ Train and validation sets have different classes!")
            issues.append(f"   Train only: {train_classes - val_classes}")
            issues.append(f"   Val only: {val_classes - train_classes}")
    
    # Check minimum samples
    if train_info:
        for class_name, info in train_info['classes'].items():
            if info['count'] < 50:
                warnings.append(f"⚠️  Class '{class_name}' has only {info['count']} training samples")
    
    # Print report
    print("\n📋 SUMMARY:")
    
    if train_info:
        print(f"   Training images: {train_info['total']}")
    if val_info:
        print(f"   Validation images: {val_info['total']}")
    if test_info:
        print(f"   Test images: {test_info['total']}")
    
    if issues:
        print("\n🚨 CRITICAL ISSUES:")
        for issue in issues:
            print(f"   {issue}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
    
    if not issues and not warnings:
        print("\n✅ Dataset looks good! Ready for training.")
    else:
        print("\n⚠️  Please fix the issues above before training.")
    
    print("=" * 60 + "\n")

# ============================================================
# MAIN
# ============================================================

def main():
    """Run complete dataset verification"""
    print("\n" + "=" * 60)
    print("X-RAY DATASET VERIFICATION TOOL")
    print("=" * 60)
    print("\nThis tool will check your dataset for:")
    print("  • Class distribution and balance")
    print("  • Duplicate images (data leakage)")
    print("  • Image quality issues")
    print("  • Proper train/val/test splits")
    print()
    
    # Analyze each split
    train_info = analyze_directory(TRAIN_DIR, "train")
    val_info = analyze_directory(VAL_DIR, "validation")
    test_info = analyze_directory(TEST_DIR, "test")
    
    # Check class balance
    check_class_balance(train_info, val_info, test_info)
    
    # Check for duplicates
    check_duplicates(train_info, val_info, test_info)
    
    # Generate final report
    generate_report(train_info, val_info, test_info)

if __name__ == "__main__":
    main()