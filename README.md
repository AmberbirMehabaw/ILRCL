
# **1. Environment Setup**

### **Python Version**

* Python **3.10+** recommended

### **Create Conda Environment**

```bash
conda create -n contrastive_env python=3.10 -y
conda activate contrastive_env
```

### **Install PyTorch (GPU + CUDA)**

Choose the appropriate CUDA version for your machine:

**CUDA 12.1**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**or CUDA 11.8**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **Install additional Python libraries**

```bash
pip install numpy matplotlib scikit-learn seaborn pillow tqdm
```

### **Dataset**

CIFAR-10 is automatically downloaded by all scripts:

```python
datasets.CIFAR10(root="./data", download=True)
```

---

# **2. Reproducing Results End-to-End**

Below are the exact commands for:

1. **Contrastive Pretraining (ResNet-18)**
2. **Contrastive Pretraining (ResNet-50)**
3. **Downstream Evaluation (Linear Classifier)**
4. **Supervised Baseline**

Commands work both locally and on Slurm (examples provided).

---

## **2.1 Contrastive Pretraining — ResNet-18**

Produces:
`simclr_resnet18_multicrop.pt`

```bash
python contrastive_pretrain_resnet18.py
```

### **Slurm example**

```bash
sbatch --job-name=pretrain18 \
       --partition=gpu_q \
       --gres=gpu:1 \
       --cpus-per-task=4 \
       --mem=16G \
       --time=48:00:00 \
       --output=pretrain18_%j.out \
       contrastive_pretrain_resnet18.py
```

---

## **2.2 Contrastive Pretraining — ResNet-50**

Produces:
`simclr_resnet50_multicrop.pt`

```bash
python contrastive_pretrain_resnet50.py
```

### **Slurm example**

```bash
sbatch --job-name=pretrain50 \
       --partition=gpu_q \
       --gres=gpu:1 \
       --cpus-per-task=4 \
       --mem=24G \
       --time=48:00:00 \
       --output=pretrain50_%j.out \
       contrastive_pretrain_resnet50.py
```

---

# **3. Downstream Evaluation (Linear Classifier)**

## **3.1 Downstream — ResNet-18 (with t-SNE, loss curves, Grad-CAM)**

Produces:

* `linear_eval_curves.png`
* `tsne_test_embeddings.png`
* `gradcam_viz/*.png`

```bash
python Downstream_resnet18.py
```

## **3.2 Downstream — ResNet-50 (with t-SNE)**

```bash
python Downstream_resnet50.py
```

### **Slurm example**

```bash
sbatch --job-name=down18 \
       --partition=gpu_q \
       --gres=gpu:1 \
       --cpus-per-task=4 \
       --mem=12G \
       --time=12:00:00 \
       Downstream_resnet18.py
```

---

# **4. Supervised Baseline Training**

Produces:

* `supervised_resnet18_multicropstyle.pt`
* Training & test accuracy/loss plots

```bash
python supervised_train.py
```

### **Slurm example**

```bash
sbatch --job-name=sup18 \
       --partition=gpu_q \
       --gres=gpu:1 \
       --cpus-per-task=4 \
       --mem=12G \
       --time=12:00:00 \
       supervised_train.py
```

---

# **5. Summary of Outputs**

| Experiment                    | Output File(s)                                                       |
| ----------------------------- | -------------------------------------------------------------------- |
| Contrastive Pretraining (R18) | `simclr_resnet18_multicrop.pt`                                       |
| Contrastive Pretraining (R50) | `simclr_resnet50_multicrop.pt`                                       |
| Downstream (R18)              | `linear_eval_curves.png`, `tsne_test_embeddings.png`, `gradcam_viz/` |
| Downstream (R50)              | `tsne_test_embeddings.png`                                           |
| Supervised                    | `supervised_resnet18_multicropstyle.pt`, training curves             |

---

If you want, I can also generate:

✅ A polished final README version
✅ A project tree layout
✅ Badges (GPU-enabled, Python version, etc.)

Just tell me!
