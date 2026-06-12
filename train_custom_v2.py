"""
Fine-tune YOLOv11 on custom single-image dataset
"""
from ultralytics import YOLO
from multiprocessing import freeze_support
import torch

def main():
    # Load your current best model (already trained on synthetic)
    model = YOLO('model/best.pt')
    
    # Fine-tune on real handwriting
    DATA_YAML = r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\custom_dataset_v2\data.yaml"
    
    print("=" * 60)
    print("🎯 FINE-TUNING ON REAL HANDWRITING DATA")
    print("=" * 60)
    
    model.train(
        data=DATA_YAML,
        epochs=30,              # Fewer epochs for fine-tuning
        lr0=0.0001,             # Lower learning rate
        patience=8,
        imgsz=640,
        batch=8,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        workers=0,
        name='finetuned_real',
        exist_ok=True
    )
    
    model.val()
    
    print("\n✅ Fine-tuning complete!")
    print('Copy with: Copy-Item "runs\\train\\finetuned_real\\weights\\best.pt" -Destination "model\\best.pt" -Force')

if __name__ == '__main__':
    freeze_support()
    main()