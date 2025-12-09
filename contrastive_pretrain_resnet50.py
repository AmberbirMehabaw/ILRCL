import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from dataclasses import dataclass

# ============================================================
# 1. Multi-Crop Transform: 2 global + 4 local (total 6 views)
# ============================================================

class MultiCropTransform:
    def __init__(self, img_size=32, num_local_crops=4):
        self.num_local = num_local_crops

        # Global crops: large random views
        self.global_t = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            ),
        ])

        # Local crops: small random views
        self.local_t = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            ),
        ])

    def __call__(self, img):
        crops = []
        # Two global crops
        crops.append(self.global_t(img))
        crops.append(self.global_t(img))
        # Local crops
        for _ in range(self.num_local):
            crops.append(self.local_t(img))
        return crops  # list of length 2 + num_local


# ============================================================
# 2. Projection Head (SimCLR v2 style)
# ============================================================

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, mode="simclrv2"):
        super().__init__()

        if mode == "linear":
            self.net = nn.Linear(in_dim, 128)

        elif mode == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
            )

        elif mode == "simclrv2":
            self.net = nn.Sequential(
                nn.Linear(in_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, 128),
            )
        else:
            raise ValueError(f"Unknown projection mode: {mode}")

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. CIFAR-10 Friendly ResNet-50 Encoder
# ============================================================

def get_encoder_resnet50_cifar():
    """
    ResNet-50 adapted for CIFAR-10 (32x32):
    - 3x3 conv, stride 1
    - no initial maxpool
    """
    m = models.resnet50(weights=None)
    m.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    m.maxpool = nn.Identity()
    features = nn.Sequential(*list(m.children())[:-1])  # up to avgpool
    dim = m.fc.in_features  # 2048
    return features, dim


# ============================================================
# 4. SimCLR Model (ResNet-50 + projector)
# ============================================================

class SimCLR(nn.Module):
    def __init__(self, proj_mode="simclrv2"):
        super().__init__()
        self.encoder, out_dim = get_encoder_resnet50_cifar()
        self.projector = ProjectionHead(out_dim, proj_mode)

    def forward(self, x):
        h = self.encoder(x)         # (B, 2048, 1, 1)
        h = torch.flatten(h, 1)     # (B, 2048)
        z = self.projector(h)       # (B, 128)
        return z


# ============================================================
# 5. Hard Negative Mining (optional)
# ============================================================

def hard_negative_mining(sim_matrix, k=5):
    """
    sim_matrix: [N, N], cosine similarity
    Returns boolean mask [N, N] where True = "hard negative".
    """
    N = sim_matrix.size(0)
    sim = sim_matrix.clone()

    # Remove self-similarity
    sim.fill_diagonal_(-1e9)

    # Top-k hardest negatives per row
    _, idx = torch.topk(sim, k=k, dim=1)
    hard_mask = torch.zeros_like(sim, dtype=torch.bool)
    hard_mask.scatter_(1, idx, True)
    return hard_mask


# ============================================================
# 6. Multi-View Contrastive Loss (2 global + 4 local)
# ============================================================

def contrastive_loss(z_list, temperature=0.2, use_hnm=False, k=5):
    """
    z_list: list of V tensors, each [B, D] (V = number of views, e.g. 6).
    We build an InfoNCE loss with multiple positives:
    - each view of a sample is a positive for all other views of the same sample.
    """
    V = len(z_list)        # number of views (e.g. 6)
    B = z_list[0].size(0)  # batch size
    N = V * B

    # Normalize & force FP32 to avoid half overflow issues
    z_list = [F.normalize(z.float(), dim=1) for z in z_list]
    z = torch.cat(z_list, dim=0)    # [N, D]

    # Similarity matrix in FP32
    sim = torch.mm(z, z.t()) / temperature  # [N, N]

    # Mask diagonal with large negative (but FP32-safe)
    diag_mask = torch.eye(N, device=sim.device, dtype=torch.bool)
    sim.masked_fill_(diag_mask, -1e4)

    # pos_mask[i, j] = True if i and j are different views of the same sample
    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    for v1 in range(V):
        for v2 in range(V):
            if v1 == v2:
                continue
            row_start = v1 * B
            col_start = v2 * B
            # place an identity matrix at [v1*B:(v1+1)*B, v2*B:(v2+1)*B]
            pos_mask[row_start:row_start+B, col_start:col_start+B] = torch.eye(
                B, device=sim.device, dtype=torch.bool
            )

    # Optional: hard negative mining reweighting
    if use_hnm:
        hnm_mask = hard_negative_mining(sim, k=k)   # [N, N] bool
        # Slightly upweight these negatives (heuristic)
        sim = sim + 0.5 * hnm_mask.float()

    # For each row i:
    #   - positives[i] = sim[i, j] for all j that are positives of i
    #   - denom[i] = log sum over all j (all similarities)
    positives = sim[pos_mask].view(N, V - 1)   # [N, V-1]
    denom = torch.logsumexp(sim, dim=1, keepdim=True)  # [N, 1]

    loss = -positives + denom  # [N, V-1]
    return loss.mean()


# ============================================================
# 7. Training Config (GPU-safe for ~11GB)
# ============================================================

@dataclass
class TrainCfg:
    proj_mode: str = "simclrv2"
    temperature: float = 0.15
    epochs: int = 400
    batch_size: int = 128           # 128 * 6 crops ≈ 768 images/effective batch
    data_dir: str = "./data"
    lr: float = 3e-4
    use_hnm: bool = True            # enable hard negative mining
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_path: str = "simclr_resnet50_multicrop.pt"


# ============================================================
# 8. Training Loop (AMP, multi-crop, hard negatives)
# ============================================================

def train():
    cfg = TrainCfg()
    print(f"Using device: {cfg.device}")

    # Dataset with multi-crop transform
    transform = MultiCropTransform(img_size=32, num_local_crops=4)
    dataset = datasets.CIFAR10(
        root=cfg.data_dir,
        train=True,
        transform=transform,
        download=True
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    model = SimCLR(proj_mode=cfg.proj_mode).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if cfg.device.startswith("cuda") else None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0

        for crops, _ in loader:
            crops = [c.to(cfg.device, non_blocking=True) for c in crops]

            # Forward in mixed precision
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    z_list = [model(c) for c in crops]
                # Loss in FP32 (contrastive_loss casts to float internally)
                loss = contrastive_loss(
                    z_list,
                    temperature=cfg.temperature,
                    use_hnm=cfg.use_hnm
                )

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z_list = [model(c) for c in crops]
                loss = contrastive_loss(
                    z_list,
                    temperature=cfg.temperature,
                    use_hnm=cfg.use_hnm
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"[Epoch {epoch:03d}/{cfg.epochs}] Loss={avg_loss:.4f}")

    # Save *just* the state_dict to be compatible with SimCLR(...) in downstream
    torch.save(model.state_dict(), cfg.out_path)
    print(f"Saved checkpoint to: {cfg.out_path}")


if __name__ == "__main__":
    train()
