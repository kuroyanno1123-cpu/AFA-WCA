"""
Step 0: AFA 無改造 1バッチ動作確認スクリプト (wandb / Lightning 不要)
  - ResNet-18 × CIFAR-10
  - train_aug = GeneralFourierOnline (AFA公式)
  - NormalJSDModule 相当のロス (CE_main + CE_aux) / 2
  - 1バッチ forward → loss.backward() が通ることを確認
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

DATA_DIR = '/home/kairisasaki/data/cifar10'
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── 1. モデル (ResNet18, AFA版) ────────────────────────────────────────────
from project.models.image_classification import get_model
model_cls = get_model('c10', 'rn18')
model = model_cls(num_classes=10).to(DEVICE)
print(f'[model] ResNet18 params: {sum(p.numel() for p in model.parameters()):,}')

# ── 2. 正規化 ──────────────────────────────────────────────────────────────
from project.dsets import get_dataset
dataset_cls = get_dataset('c10')
normalise = T.Normalize(mean=dataset_cls.mean, std=dataset_cls.std)

# ── 3. train_aug = GeneralFourierOnline (AFA公式) ─────────────────────────
from project.augs.fba.fourier_basis import GeneralFourierOnline
groups  = list(range(1, 16))   # AFA公式デフォルトに近い設定
phases  = (0, 1)
fba_aug = GeneralFourierOnline(
    img_size=32, groups=groups, phases=phases,
    f_cut=1, phase_cut=1, min_str=0, mean_str=5, granularity=64
)
print(f'[aug]   GeneralFourierOnline: groups={groups[:3]}..., phases={phases}')

# ── 4. データセット (ToTensor のみ → FBAに渡す) ───────────────────────────
from project.dsets.vision.dataset import CIFAR10Dataset
train_transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
t_dset = CIFAR10Dataset(root=DATA_DIR, train=True, transform=train_transform, download=False)
t_dl   = DataLoader(t_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ── 5. 1バッチ取得 ─────────────────────────────────────────────────────────
x, y = next(iter(t_dl))
x, y = x.to(DEVICE), y.to(DEVICE)
print(f'[data]  x.shape={x.shape}, x.dtype={x.dtype}, range=[{x.min():.3f}, {x.max():.3f}]')

# ── 6. 補助ストリーム生成 (FBA) ────────────────────────────────────────────
x_aux = fba_aug(x)  # Tensor[0,1] → Tensor[0,1]
print(f'[aug]   x_aux.shape={x_aux.shape}, range=[{x_aux.min():.3f}, {x_aux.max():.3f}]')

# ── 7. forward (2ストリーム) ──────────────────────────────────────────────
model.train()
logits_main = model(normalise(x))
logits_aux  = model(normalise(x_aux))

# ── 8. ACE ロス = (CE_main + CE_aux) / 2 ─────────────────────────────────
ce_main = F.cross_entropy(logits_main, y)
ce_aux  = F.cross_entropy(logits_aux,  y)
loss    = (ce_main + ce_aux) / 2.0

print(f'[loss]  CE_main={ce_main.item():.4f}  CE_aux={ce_aux.item():.4f}  ACE={loss.item():.4f}')

# ── 9. backward ───────────────────────────────────────────────────────────
loss.backward()
grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
print(f'[grad]  grad_norm={grad_norm:.4f}')

print('\n✓ Step 0 PASS: AFA 無改造 1バッチ forward/backward 完了')
print(f'  device={DEVICE}  batch_size={BATCH_SIZE}')
print(f'  loss={loss.item():.4f}  grad_norm={grad_norm:.4f}')
