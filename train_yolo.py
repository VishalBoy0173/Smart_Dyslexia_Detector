from ultralytics import YOLO

# 1. Load a pre-trained YOLOv11 model (nano version – fast & lightweight)
model = YOLO('yolo11n.pt')   # You can also try yolo11s, yolo11m for potentially higher accuracy

# 2. Fine-tune on our dyslexia dataset
model.train(
    data='synthetic_dyslexia_dataset/data.yaml',   # path to your YAML config
    epochs=100,          # maximum number of epochs
    patience=15,         # early stopping if no improvement in 15 epochs
    imgsz=640,           # image size
    batch=16,            # reduce if you run out of memory (e.g., 8)
    name='dyslexia_detector',
    exist_ok=True        # overwrite existing results folder
)

# 3. Validate the trained model
model.val()

# 4. Export (the best weights are automatically saved as best.pt)
#    The file will be at: runs/train/dyslexia_detector/weights/best.pt
print("✅ Training complete. Copy best.pt to the 'model/' folder.")