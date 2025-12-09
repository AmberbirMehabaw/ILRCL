import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from dataclasses import dataclass

# ============================================================
# 1. Multi-Crop (2 global + 4 local) — fits 11GB GPU
# ============================================================

class MultiCropTransform:
    def __init__(self, img_size=32, num_local_crops=4):
        self.num_local = num_local_crops

        self.global_t = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(0.2),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
        ])

        self.local_t = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
        ])

    def __call__(self, img):
        crops = [self.global_t(img), self.global_t(img)]
        crops += [self.local_t(img) for _ in range(self.num_local)]
        return crops


# ============================================================
# 2. Projection Head Variants
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
                nn.Linear(512, 128)
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

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. ResNet-18 Encoder (CIFAR-friendly)
# ============================================================

def get_encoder():
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    features = nn.Sequential(*list(m.children())[:-1])
    dim = m.fc.in_features
    return features, dim


# ============================================================
# 4. SimCLR Model With Options
# ============================================================

class SimCLR(nn.Module):
    def __init__(self, proj_mode="simclrv2"):
        super().__init__()
        self.encoder, out_dim = get_encoder()
        self.projector = ProjectionHead(out_dim, proj_mode)

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        z = self.projector(h)
        return z


# ============================================================
# 5. Hard Negative Mining (optional)
# ============================================================

def hard_negative_mining(sim_matrix, k=5):
    """
    Picks top-k hardest negatives for each sample.
    Returns: mask of shape [N, N]
    """
    N = sim_matrix.size(0)
    sim = sim_matrix.clone()

    # mask out positives
    sim.fill_diagonal_(-1e9)

    # get top-k negatives
    _, idx = torch.topk(sim, k=k, dim=1)
    hard_mask = torch.zeros_like(sim, dtype=torch.bool)
    hard_mask.scatter_(1, idx, True)
    return hard_mask


# ============================================================
# 6. Contrastive Loss with Multi-Crop + Hard Negatives
# ============================================================

def contrastive_loss(z_list, temperature=0.2, use_hnm=False, k=5):
    V = len(z_list)  # views
    B = z_list[0].shape[0]
    N = V * B

    # Normalize & cast to float32 explicitly
    z_list = [F.normalize(z.float(), dim=1) for z in z_list]
    z = torch.cat(z_list, dim=0)  # [N, D] in float32

    sim = torch.mm(z, z.t()) / temperature  # float32

    # mask diag with a big but FP32-safe negative
    mask = torch.eye(N, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e4)   # -1e4 is plenty to kill logits

    # positive matrix
    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    for i in range(V):
        for j in range(V):
            if i != j:
                pos_mask[i*B:(i+1)*B, j*B:(j+1)*B] = torch.eye(B, device=z.device, dtype=torch.bool)

    positives = sim[pos_mask].view(N, V-1)
    denom = torch.logsumexp(sim, dim=1).unsqueeze(1)

    if use_hnm:
        hnm_mask = hard_negative_mining(sim, k)
        sim = sim + 0.5 * hnm_mask.float()

    loss = -positives + denom
    loss = loss.mean()
    return loss



# ============================================================
# 7. Training Config (GPU Safe)
# ============================================================

@dataclass
class TrainCfg:
    proj_mode: str = "simclrv2"
    temperature: float = 0.15
    epochs: int = 500
    batch_size: int = 128            # fits 11GB GPU
    data_dir: str = "./data"
    lr: float = 3e-4
    use_hnm: bool = True             # enable hard negative mining
    device: str = "cuda"


# ============================================================
# 8. Training Loop (AMP Enabled)
# ============================================================

def train():
    cfg = TrainCfg()

    dataset = datasets.CIFAR10(
        root=cfg.data_dir,
        train=True,
        transform=MultiCropTransform(img_size=32),
        download=True
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    model = SimCLR(cfg.proj_mode).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(cfg.epochs):
        model.train()
        for crops, _ in loader:
            crops = [c.to(cfg.device) for c in crops]
        
            # 1) Forward in AMP (new API)
            with torch.amp.autocast("cuda"):
                z_list = [model(c) for c in crops]
        
            # 2) Loss in full precision (no AMP here)
            loss = contrastive_loss(
                z_list,
                cfg.temperature,
                use_hnm=cfg.use_hnm
            )
        
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()


        print(f"Epoch {epoch+1}/{cfg.epochs} Loss={loss.item():.4f}")

    torch.save(model.state_dict(), "simclr_resnet18_multicrop.pt")
    print("Saved checkpoint.")


if __name__ == "__main__":
    train()
