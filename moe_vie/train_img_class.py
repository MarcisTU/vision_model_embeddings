import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from src.open_clip import create_model_and_transforms, image_to_device


class MoEViEImageDataset(Dataset):
    def __init__(self, image_paths, labels, preprocess):
        """
        image_paths: List[str] - paths to image files
        labels: List[int] - integer class labels
        preprocess: OpenCLIP preprocess function
        """
        self.image_paths = image_paths
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(path).convert("RGB")
        processed_image = self.preprocess(image)
        
        return processed_image, label


class MoEViECollate:
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, batch):
        images, labels = zip(*batch)
        
        # Prepare format required by MoEViE collate_fn
        tuples = [(img, 0) for img in images]
        packed, _ = self.preprocess.collate_fn(tuples)
        
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return packed, labels_tensor


class MoEViEClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        # MoEViE-B16 output visual embedding dim is 1024
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, packed_images):
        # Encode visual features
        image_features = self.clip_model.encode_image(packed_images, normalize=True)
        logits = self.classifier(image_features)
        return logits


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, mean, std, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for packed_images, labels in pbar:
        # Move packed images and labels to GPU using MoEViE helper
        packed_images = image_to_device(packed_images, device, torch.float32, mean=mean, std=std)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed Precision Training
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(packed_images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, mean, std, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Validation")
    for packed_images, labels in pbar:
        packed_images = image_to_device(packed_images, device, torch.float32, mean=mean, std=std)
        labels = labels.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(packed_images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"val_loss": loss.item(), "val_acc": correct / total})

    return running_loss / total, correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    base_model, _, preprocess = create_model_and_transforms(
        "MoEViE-B16-224",
        pretrained=True,
        force_preprocess_cfg=dict(
            patch_size=16, size_range=(224, 224), center_crop=True, window_size=1
        ),
        image_mean=MEAN, image_std=STD,
        device=device
    )

    # Dummy Dataset Setup (Replace with your actual paths and labels)
    train_paths = ["./cat.png"] * 32
    train_labels = [0] * 32
    val_paths = ["./cat.png"] * 16
    val_labels = [0] * 16
    num_classes = 3

    train_dataset = MoEViEImageDataset(train_paths, train_labels, preprocess)
    val_dataset = MoEViEImageDataset(val_paths, val_labels, preprocess)

    collate_fn = MoEViECollate(preprocess)

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    # Initialize Classifier Wrapper
    model = MoEViEClassifier(base_model, num_classes=num_classes).to(device)

    # only train classification head
    for param in model.clip_model.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler = torch.amp.GradScaler()

    num_epochs = 5
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, MEAN, STD, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, MEAN, STD, device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save Best Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_moevie_model.pth")
            print("Saved new best model checkpoint!")


if __name__ == "__main__":
    main()
