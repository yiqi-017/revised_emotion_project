import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloaders
from model import EmotionCNN


BASE_DIR = Path(__file__).resolve().parent


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

    if total == 0:
        raise ValueError("Validation loader returned zero samples.")

    return total_loss / total, correct / total


def train():
    set_seed(42)

    train_dir = BASE_DIR / "DATASET" / "train"
    val_dir = BASE_DIR / "DATASET" / "test"
    batch_size = 32
    num_epochs = 10
    lr = 1e-4
    weight_decay = 1e-4
    num_workers = 0
    use_weighted_sampler = True
    save_path = BASE_DIR / "checkpoints" / "emotion_model.pth"

    os.makedirs(BASE_DIR / "checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, classes, class_to_idx = get_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampler=use_weighted_sampler,
    )
    print("Classes:", classes)
    print("Class to idx:", class_to_idx)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Using device: {device}")

    model = EmotionCNN(num_classes=len(classes), pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size_now = labels.size(0)
            running_loss += loss.item() * batch_size_now
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size_now

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

        if total == 0:
            raise ValueError("Training loader returned zero samples.")

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "class_to_idx": class_to_idx,
                    "image_size": 224,
                },
                save_path,
            )
            print(f"Best model saved to {Path(save_path).resolve()}")

    print(f"Training finished. Best val acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
