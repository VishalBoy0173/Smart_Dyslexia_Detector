"""
check_model.py - Check model accuracy and metrics
"""

from ultralytics import YOLO
import os

# ✅ CORRECTED PATHS FOR C:\Users\visha\FYP\ LOCATION
MODEL_PATH = r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\model\best.pt"
DATA_YAML = r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\synthetic_dyslexia_dataset\data.yaml"

print("=" * 60)
print("📊 MODEL PERFORMANCE REPORT")
print("=" * 60)

print("\n1. Checking files:")
print(f"   Model: {MODEL_PATH} -> {os.path.exists(MODEL_PATH)}")
print(f"   Data: {DATA_YAML} -> {os.path.exists(DATA_YAML)}")

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found!")
    exit()

if not os.path.exists(DATA_YAML):
    print("❌ data.yaml not found!")
    exit()

print("\n2. data.yaml content:")
with open(DATA_YAML, 'r') as f:
    print(f.read())

print("\n3. Loading model...")
model = YOLO(MODEL_PATH)

print("\n4. Validating on Synthetic Dataset...")
metrics = model.val(data=DATA_YAML)

print(f"\n   📊 Synthetic Dataset Results:")
print(f"   mAP@0.5:      {metrics.box.map50:.4f}")
print(f"   mAP@0.5-0.95: {metrics.box.map:.4f}")
print(f"   Precision:    {metrics.box.mp:.4f}")
print(f"   Recall:       {metrics.box.mr:.4f}")

# Model info
print(f"\n5. Model Information:")
print(f"   Model path:    {MODEL_PATH}")
print(f"   File size:     {os.path.getsize(MODEL_PATH) / 1024:.1f} KB")
print(f"   Classes:       {model.names}")

print("\n" + "=" * 60)
print("✅ Model performance check complete!")
print("=" * 60)