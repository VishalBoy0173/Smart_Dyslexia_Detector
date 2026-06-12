"""
Check model accuracy and metrics
"""
from ultralytics import YOLO

# Load your model
model = YOLO('model/best.pt')

print("=" * 60)
print("📊 MODEL PERFORMANCE REPORT")
print("=" * 60)

# Validate on the synthetic dataset
print("\n1. Validating on Synthetic Dataset...")
synthetic_yaml = r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\synthetic_dyslexia_dataset\data.yaml"

import os
if os.path.exists(synthetic_yaml):
    metrics = model.val(data=synthetic_yaml)
    
    print(f"\n   📊 Synthetic Dataset Results:")
    print(f"   mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"   mAP@0.5-0.95: {metrics.box.map:.4f}")
    print(f"   Precision:    {metrics.box.mp:.4f}")
    print(f"   Recall:       {metrics.box.mr:.4f}")
else:
    print("   Synthetic dataset not found")

# Model info
print(f"\n2. Model Information:")
print(f"   Model path:    model/best.pt")
print(f"   File size:     {os.path.getsize('model/best.pt') / 1024:.1f} KB")
print(f"   Classes:       {model.names}")
print(f"   Parameters:    2,582,737")
print(f"   GFLOPs:        6.3")

# Quick test on a single image
print(f"\n3. Quick Inference Test:")
import cv2
import numpy as np

# Create a simple test
test_img = np.zeros((64, 64, 3), dtype=np.uint8)
cv2.imwrite('test_letter.png', test_img)

results = model('test_letter.png', conf=0.25, iou=0.5, imgsz=64)
speed = results[0].speed if results[0].speed else {}
print(f"   Inference speed: {speed.get('inference', 'N/A')} ms")

if os.path.exists('test_letter.png'):
    os.remove('test_letter.png')

print("\n" + "=" * 60)
print("✅ Model performance check complete!")
print("=" * 60)