"""
Convert single-image dataset (control/dyslexic/corrected) to YOLOv11 format.
Each image is one word/sentence with its class label.
"""
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# ═══════════ CONFIG ═══════════
INPUT_PATH = r"C:\Users\visha\FYP\data_singleImage"
OUTPUT_PATH = r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\custom_dataset_v2"

CLASS_MAP = {
    'control': 0,      # Normal
    'dyslexic': 1,     # Reversal
    'corrected': 2     # Corrected
}

def find_letters_and_create_labels(image_path, class_id, output_img_dir, output_label_dir, prefix):
    """Find individual letters and create YOLO labels"""
    img = cv2.imread(image_path)
    if img is None:
        return 0
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    # Create binary for contour finding
    if avg_brightness > 127:
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    else:
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort contours
    valid_contours = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw >= 8 and ch >= 8:
            if cw < w * 0.9 or ch < h * 0.9:
                valid_contours.append((x, y, cw, ch))
    
    valid_contours.sort(key=lambda b: (b[1] // 30, b[0]))
    
    if not valid_contours:
        return 0
    
    # Save image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_name = f"{prefix}_{base_name}"
    output_img_path = os.path.join(output_img_dir, output_name + '.jpg')
    cv2.imwrite(output_img_path, img)
    
    # Create YOLO label file
    label_path = os.path.join(output_label_dir, output_name + '.txt')
    with open(label_path, 'w') as f:
        for x, y, cw, ch in valid_contours:
            x_center = (x + cw/2) / w
            y_center = (y + ch/2) / h
            norm_w = cw / w
            norm_h = ch / h
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    
    return 1


def process_dataset():
    """Main conversion function"""
    print("=" * 60)
    print("🔄 CONVERTING SINGLE-IMAGE DATASET TO YOLO FORMAT")
    print("=" * 60)
    
    # Create output folders
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_PATH, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, split, 'labels'), exist_ok=True)
    
    total_processed = 0
    
    for split_name in ['Train', 'Test']:
        split_input = os.path.join(INPUT_PATH, split_name)
        split_output = 'train' if split_name == 'Train' else 'val'
        
        print(f"\n📂 Processing {split_name} → {split_output}...")
        
        for class_name, class_id in CLASS_MAP.items():
            class_path = os.path.join(split_input, class_name)
            
            if not os.path.exists(class_path):
                print(f"  ⚠️ Not found: {class_path}")
                continue
            
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            count = 0
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                result = find_letters_and_create_labels(
                    img_path, class_id,
                    os.path.join(OUTPUT_PATH, split_output, 'images'),
                    os.path.join(OUTPUT_PATH, split_output, 'labels'),
                    f"{class_name}"
                )
                count += result
            
            print(f"  ✅ {class_name}: {count}/{len(image_files)} images")
            total_processed += count
    
    # Create data.yaml
    yaml_path = os.path.join(OUTPUT_PATH, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"""path: {OUTPUT_PATH}
train: train/images
val: val/images
nc: 3
names:
  0: normal
  1: reversal
  2: corrected
""")
    
    print(f"\n✅ Conversion complete!")
    print(f"📊 Total images processed: {total_processed}")
    print(f"📁 Output: {OUTPUT_PATH}")
    print(f"📄 data.yaml created")
    print(f"\nNext: python train_custom_v2.py")


if __name__ == "__main__":
    process_dataset()