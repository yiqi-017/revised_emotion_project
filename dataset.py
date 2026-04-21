from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RAFDB_CLASS_NAMES = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happy",
    "5": "sad",
    "6": "angry",
    "7": "neutral",
}


def build_transforms(image_size: int = IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, val_transform


def validate_imagefolder_dir(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {p.resolve()}")
    if not any(child.is_dir() for child in p.iterdir()):
        raise ValueError(
            f"Dataset path {p.resolve()} does not contain class subfolders. "
            "Expected ImageFolder format like data/train/happy/*.jpg"
        )


def map_class_names(folder_classes: List[str]) -> List[str]:
    return [RAFDB_CLASS_NAMES.get(class_name, class_name) for class_name in folder_classes]


def get_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, List[str], dict]:
    validate_imagefolder_dir(train_dir)
    validate_imagefolder_dir(val_dir)

    train_transform, val_transform = build_transforms()
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty.")
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Training and validation datasets have different class folders: "
            f"train={train_dataset.classes}, val={val_dataset.classes}"
        )

    sampler = None
    shuffle = True
    class_weights = None

    if use_weighted_sampler:
        targets = torch.tensor(train_dataset.targets)
        class_counts = torch.bincount(targets)
        class_weights = 1.0 / class_counts.float().clamp(min=1)
        sample_weights = class_weights[targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    classes = map_class_names(train_dataset.classes)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    return train_loader, val_loader, classes, class_to_idx
