import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import os
import time

# ======================================================
# CUDA OPTIMIZATION SETTINGS
# ======================================================
torch.backends.cudnn.benchmark = True  # Faster for fixed input size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================================================
# PATHS
# ======================================================
DATA_DIR = r"C:\Users\akshi\Documents\AI Crop Proj\data\crop_disease\PlantVillage_Color"
SAVE_DIR = r"C:\Users\akshi\Documents\AI Crop Proj\models\crop_disease"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# HYPERPARAMETERS
# ======================================================
IMG_SIZE = 224
BATCH_SIZE = 64          # Increased for GPU
EPOCHS = 12
LR = 0.0003
NUM_WORKERS = 0

# ======================================================
# TRANSFORMS
# ======================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================================================
# DATASET
# ======================================================
print("Loading dataset...")

full_dataset = datasets.ImageFolder(DATA_DIR)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

num_classes = len(full_dataset.classes)
print("Classes:", num_classes)
print("Total Images:", len(full_dataset))

# ======================================================
# MODEL
# ======================================================
print("Loading MobileNetV2...")

model = models.mobilenet_v2(weights="IMAGENET1K_V1")

# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# ======================================================
# LOSS, OPTIMIZER, AMP
# ======================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

scaler = GradScaler()  # Mixed precision scaler

# ======================================================
# TRAINING LOOP
# ======================================================
best_acc = 0
total_start_time = time.time()

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    epoch_start = time.time()
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # ===== Mixed Precision Forward =====
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # ===== Backward =====
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # ================= VALIDATION =================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    epoch_time = (time.time() - epoch_start) / 60

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {train_loss:.4f} | "
          f"Val Acc: {val_acc:.2f}% | "
          f"Time: {epoch_time:.2f} min")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, "best_model.pth"))

print("\nTraining Complete")
print("Best Validation Accuracy:", best_acc)
print("Total Training Time: {:.2f} min".format((time.time() - total_start_time)/60))
