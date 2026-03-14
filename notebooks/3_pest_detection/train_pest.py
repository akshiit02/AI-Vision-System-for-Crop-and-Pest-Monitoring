import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from collections import Counter
import os
import time

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================================================
# PATHS
# ======================================================
DATA_DIR = "../../data/pest/farm_insects"
SAVE_DIR = "../../models/pest"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# HYPERPARAMETERS
# ======================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4

# ======================================================
# TRANSFORMS (Stronger Augmentation)
# ======================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(20, translate=(0.1, 0.1)),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ======================================================
# LOAD DATASET
# ======================================================
print("Loading dataset...")

full_dataset = datasets.ImageFolder(DATA_DIR)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

num_classes = len(full_dataset.classes)
print("Classes:", num_classes)
print("Total Images:", len(full_dataset))

# ======================================================
# CLASS WEIGHTS (Handle Imbalance)
# ======================================================
labels = [label for _, label in full_dataset.samples]
class_counts = Counter(labels)

weights = []
for i in range(num_classes):
    weights.append(1.0 / class_counts[i])

class_weights = torch.tensor(weights, dtype=torch.float).to(device)

# ======================================================
# MODEL (EfficientNet Full Fine-Tune)
# ======================================================
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Unfreeze entire network
for param in model.parameters():
    param.requires_grad = True

model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# ======================================================
# LOSS + OPTIMIZER
# ======================================================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# ======================================================
# TRAINING LOOP
# ======================================================
best_acc = 0
start_time = time.time()

print("\nStarting fine-tuning...\n")

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # ================= VALIDATION =================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {train_loss:.4f} | "
          f"Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, "best_model.pth"))

print("\nTraining Complete")
print("Best Validation Accuracy:", best_acc)
print("Total Time: {:.2f} min".format((time.time() - start_time)/60))
