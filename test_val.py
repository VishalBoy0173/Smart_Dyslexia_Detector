from ultralytics import YOLO
model = YOLO(r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\model\best.pt")
metrics = model.val(data=r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\synthetic_dyslexia_dataset\data.yaml")
print(metrics.box.map50, metrics.box.mp, metrics.box.mr)