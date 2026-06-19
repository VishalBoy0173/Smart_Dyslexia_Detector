"""
DyslexiAI – 3‑Class Handwriting Classification (Normal / Reversal / Corrected)
Trains on custom dataset with images of size ~28×29.
Uses GPU acceleration and early stopping.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')



# -------------------------------
# 1. Configuration
# -------------------------------
DATA_PATH = r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\Dataset Dyslexia_Password WanAsy321\Gambo"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
IMAGE_SIZE = 64          # Resize small images to 64x64 for better CNN performance
NUM_CLASSES = 3
CLASS_NAMES = ['Normal', 'Reversal', 'Corrected']
PATIENCE = 10            # Early stopping patience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if device.type == "cuda":
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ CUDA not available – training will be slow. Install CUDA toolkit and PyTorch with CUDA support.")

# -------------------------------
# 2. Helper to find train/test folders (case‑insensitive)
# -------------------------------
def find_folder(root, possible_names):
    for name in possible_names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return None

# -------------------------------
# 3. Custom Dataset
# -------------------------------
class DyslexiaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {'Normal': 0, 'Reversal': 1, 'Corrected': 2}
        
        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"⚠️ Warning: {class_dir} not found – skipping {class_name}.")
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.images.append(os.path.join(class_dir, fname))
                    self.labels.append(label)
        print(f"📁 Loaded {len(self.images)} images from {root_dir}")
        for name, idx in self.class_to_idx.items():
            cnt = self.labels.count(idx)
            print(f"   - {name}: {cnt} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------------------
# 4. Data Transforms
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# 5. CNN Model (suitable for 64x64 input)
# -------------------------------
class DyslexiaCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(DyslexiaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # After 4 pooling layers: 64 -> 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# -------------------------------
# 6. Training & Evaluation Functions
# -------------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return epoch_loss, acc, prec, rec, f1, all_preds, all_labels

# -------------------------------
# 7. Main Training Pipeline
# -------------------------------
def main():
    print("\n" + "="*70)
    print("DYSLEXIA DETECTION – 3‑CLASS TRAINING")
    print("="*70)
    
    # Locate train and test folders (supports Train/train and Test/test)
    train_dir = find_folder(DATA_PATH, ['Train', 'train'])
    test_dir = find_folder(DATA_PATH, ['Test', 'test'])
    
    if train_dir is None:
        print(f"❌ No train folder found in {DATA_PATH}. Expected 'Train' or 'train'.")
        return
    if test_dir is None:
        print(f"❌ No test folder found in {DATA_PATH}. Expected 'Test' or 'test'.")
        return
    
    print(f"📂 Using train folder: {train_dir}")
    print(f"📂 Using test folder: {test_dir}")
    
    # Load datasets
    full_train = DyslexiaDataset(train_dir, transform=train_transform)
    test_dataset = DyslexiaDataset(test_dir, transform=val_transform)
    
    # Split training into train/validation (90/10)
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    # Use validation transform
    val_dataset.dataset.transform = val_transform
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"\n📊 Data splits:")
    print(f"   Training: {len(train_dataset)} images")
    print(f"   Validation: {len(val_dataset)} images")
    print(f"   Test: {len(test_dataset)} images")
    
    # Model, loss, optimizer
    model = DyslexiaCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop with early stopping
    best_val_f1 = 0.0
    counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    print("\n🚀 Starting training...\n")
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model_3class.pth")
            print(f"   ✅ New best model saved (val_f1={val_f1:.4f})")
            counter = 0
        else:
            counter += 1
        
        # Print progress every 5 epochs
        if epoch % 5 == 0:
            print(f"\nEpoch {epoch}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if counter >= PATIENCE:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}")
            break
    
    # Load best model and evaluate on test set
    if os.path.exists("best_model_3class.pth"):
        model.load_state_dict(torch.load("best_model_3class.pth"))
    else:
        print("⚠️ No best model found – using current model.")
    
    test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion)
    
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix – 3‑Class Classification')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Training curves
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    # Save final model for deployment
    os.makedirs("backend/models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': CLASS_NAMES,
        'input_size': (3, IMAGE_SIZE, IMAGE_SIZE)
    }, "backend/models/dyslexia_3class.pth")
    
    print("\n✅ Training complete!")
    print("📁 Best model saved as 'best_model_3class.pth'")
    print("📁 Deployment model saved as 'backend/models/dyslexia_3class.pth'")

if __name__ == "__main__":
    main()