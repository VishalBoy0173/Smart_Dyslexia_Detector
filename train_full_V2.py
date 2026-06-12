"""
Complete Two-Stage Training for Dyslexia Detection
Stage 1: Train on synthetic dataset (50 epochs)
Stage 2: Fine-tune on custom real handwriting dataset (30 epochs)
Saves the best model to model/best.pt automatically
"""
from ultralytics import YOLO
from multiprocessing import freeze_support
import torch
import os
import shutil

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("🚀 SMART DYSLEXIA DETECTOR - MODEL TRAINING")
    print("=" * 60)
    print(f"🖥️  Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")    
    # ═══════════════════════════════════════════
    # STAGE 1: Train on Synthetic Dataset
    # ═══════════════════════════════════════════
    synthetic_yaml = r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\synthetic_dyslexia_dataset\data.yaml"
    
    if os.path.exists(synthetic_yaml):
        print("\n" + "=" * 60)
        print("📚 STAGE 1: Training on Synthetic Dyslexia Dataset")
        print("=" * 60)
        
        model = YOLO('yolo11n.pt')
        
        results = model.train(
            data=synthetic_yaml,
            epochs=50,
            patience=10,
            imgsz=640,
            batch=16,
            device=device,
            workers=0,
            name='stage1_synthetic',
            exist_ok=True
        )
        
        stage1_path = 'runs/train/stage1_synthetic/weights/best.pt'
        
        # Validate stage 1
        model = YOLO(stage1_path)
        metrics1 = model.val()
        print(f"\n📊 Stage 1 Results:")
        print(f"   mAP@0.5:      {metrics1.box.map50:.4f}")
        print(f"   mAP@0.5-0.95: {metrics1.box.map:.4f}")
    else:
        print(f"❌ Synthetic dataset not found at: {synthetic_yaml}")
        print("   Using base yolo11n.pt as fallback.")
        stage1_path = 'yolo11n.pt'
    
    # ═══════════════════════════════════════════
    # STAGE 2: Fine-Tune on Custom Dataset
    # ═══════════════════════════════════════════
    custom_yaml = r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\custom_dataset\data.yaml"
    
    if os.path.exists(custom_yaml):
        print("\n" + "=" * 60)
        print("🎯 STAGE 2: Fine-Tuning on Custom Handwriting Dataset")
        print("=" * 60)
        
        model = YOLO(stage1_path)
        
        results = model.train(
            data=custom_yaml,
            epochs=30,
            lr0=0.0001,
            patience=8,
            imgsz=640,
            batch=8,
            device=device,
            workers=0,
            name='stage2_finetuned',
            exist_ok=True,
            hsv_h=0.015,
            hsv_s=0.3,
            hsv_v=0.2,
            degrees=3.0,
            translate=0.1,
            scale=0.3,
            mosaic=0.3,
        )
        
        final_path = 'runs/train/stage2_finetuned/weights/best.pt'
        
        # Validate final model
        model = YOLO(final_path)
        metrics2 = model.val()
        print(f"\n📊 Stage 2 (Final) Results:")
        print(f"   mAP@0.5:      {metrics2.box.map50:.4f}")
        print(f"   mAP@0.5-0.95: {metrics2.box.map:.4f}")
        print(f"   Classes:      {model.names}")
    else:
        print(f"\n⚠️  Custom dataset not found at: {custom_yaml}")
        print("   Run convert_dataset_to_yolo.py first if you have real handwriting images.")
        print("   Using Stage 1 model as final model.")
        final_path = stage1_path
    
    # ═══════════════════════════════════════════
    # COPY FINAL MODEL TO APP
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📦 COPYING MODEL TO APPLICATION")
    print("=" * 60)
    
    dest_path = r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\model\best.pt"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy(final_path, dest_path)
    
    print(f"✅ Model copied to: {dest_path}")
    
    # Verify
    model = YOLO(dest_path)
    print(f"✅ Model verified! Classes: {model.names}")
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print("Run your app with: python app.py")
    print("=" * 60)

if __name__ == '__main__':
    freeze_support()
    main()