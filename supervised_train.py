# supervised_train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from dataclasses import dataclass
import matplotlib.pyplot as plt


# ============================================================
# 1. CIFAR-friendly ResNet-18
# ============================================================

def get_encoder():
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    features = nn.Sequential(*list(m.children())[:-1])
    dim = m.fc.in_features
    return features, dim


# ============================================================
# 2. Same augmentation as SimCLR
# ============================================================

def get_supervised_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(0.2),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2470,0.2435,0.2616)),
    ])


# ============================================================
# 3. Full supervised classifier (encoder + FC)
# ============================================================

class SupModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder, dim = get_encoder()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        return self.fc(h)


# ============================================================
# 4. Config
# ============================================================

@dataclass
class TrainCfg:
    epochs: int = 200
    lr: float = 3e-4
    batch_size: int = 128
    data_dir: str = "./data"
    device: str = "cuda"


# ============================================================
# 5. Evaluation — returns loss + accuracy
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            loss = F.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()

            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# ============================================================
# 6. Training loop (with metric logging + plots)
# ============================================================

def supervised_train():
    cfg = TrainCfg()

    transform = get_supervised_transform()

    train_ds = datasets.CIFAR10(root=cfg.data_dir, train=True,
                                transform=transform, download=True)
    test_ds  = datasets.CIFAR10(root=cfg.data_dir, train=False,
                                transform=transform, download=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=256,
                             shuffle=False, num_workers=4)

    model = SupModel().to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler()

    # metric logs
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    best_acc = 0

    for epoch in range(1, cfg.epochs + 1):

        # -------- TRAIN --------
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)

            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # -------- EVAL --------
        train_loss, train_acc = evaluate(model, train_loader, cfg.device)
        test_loss, test_acc = evaluate(model, test_loader, cfg.device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f} "
                  f"TestLoss={test_loss:.4f} TestAcc={test_acc:.4f}")

        best_acc = max(best_acc, test_acc)

    print(f"\n*** BEST SUPERVISED ACCURACY = {best_acc:.4f} ***")

    # Save checkpoint
    torch.save(model.state_dict(), "supervised_resnet18_multicropstyle.pt")

    # ---------------- PLOTS ----------------
    # LOSS CURVE
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Supervised Train/Test Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig("supervised_loss_curve.png", dpi=200)
    plt.close()

    # ACCURACY CURVE
    plt.figure(figsize=(8,5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Supervised Train/Test Accuracy Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig("supervised_accuracy_curve.png", dpi=200)
    plt.close()

    print("Saved supervised_loss_curve.png and supervised_accuracy_curve.png")


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    supervised_train()
