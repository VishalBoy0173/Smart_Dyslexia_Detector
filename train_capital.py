# =========================
# 1. IMPORT LIBRARIES
# =========================
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# =========================
# 2. DATASET PATH (YOUR LOCAL PATH)
# =========================
DATASET_PATH = r"C:\Users\visha\FYP\Capital Dataset\archive\Dyslexia Handwriting dataset"

train_dir = os.path.join(DATASET_PATH, "train")
valid_dir = os.path.join(DATASET_PATH, "valid")
test_dir  = os.path.join(DATASET_PATH, "test")

# Verify paths
print("\n📁 Checking dataset paths:")
print(f"   Train: {train_dir} — Exists: {os.path.exists(train_dir)}")
print(f"   Valid: {valid_dir} — Exists: {os.path.exists(valid_dir)}")
print(f"   Test:  {test_dir}  — Exists: {os.path.exists(test_dir)}")

# Check folder contents
if os.path.exists(train_dir):
    folders = os.listdir(train_dir)
    print(f"\n📂 Train folders ({len(folders)}): {sorted(folders)[:5]}...")

# =========================
# 3. PARAMETERS
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =========================
# 4. DATA AUGMENTATION
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

# =========================
# 5. LOAD DATA
# =========================
print("\n📥 Loading training data...")
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print("\n📥 Loading validation data...")
valid_data = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("\n📥 Loading test data...")
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_data.class_indices)
print(f"\n🏷  Classes ({num_classes}): {train_data.class_indices}")

# =========================
# 6. BUILD MODEL
# =========================
print("\n🏗️  Building model...")
base_model = EfficientNetB0(
    include_top=False,
    weights=None,              # No internet needed
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# =========================
# 7. COMPILE MODEL
# =========================
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# 8. TRAIN MODEL (STAGE 1)
# =========================
EPOCHS = 10

print(f"\n🚀 Starting Stage 1 Training ({EPOCHS} epochs)...")
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS
)

# =========================
# 9. FINE TUNING (STAGE 2)
# =========================
print("\n🔧 Starting Stage 2 Fine-Tuning...")
base_model.trainable = True

# Freeze early layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

FINE_EPOCHS = 5

history_fine = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=FINE_EPOCHS
)

# =========================
# 10. EVALUATE ON TEST DATA
# =========================
print("\n📊 Evaluating on test data...")
test_loss, test_acc = model.evaluate(test_data)
print(f"   Test Accuracy: {test_acc:.4f}")
print(f"   Test Loss: {test_loss:.4f}")

# =========================
# 11. SAVE MODEL
# =========================
save_path = r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\model\efficientnet_capital.h5"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)
print(f"\n✅ Model saved to: {save_path}")

# =========================
# 12. PLOT ACCURACY
# =========================
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.axvline(x=EPOCHS, color='red', linestyle='--', alpha=0.5, label='Fine-tune start')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.axvline(x=EPOCHS, color='red', linestyle='--', alpha=0.5, label='Fine-tune start')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\model\training_plot.png")
plt.show()

print("\n🎉 Training complete!")