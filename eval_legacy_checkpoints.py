"""
旧 train_comparison_v3.py で学習した best.pt を 19 種 CIFAR-10-C で再評価。
モデル: rn18（標準、DuBN なし）
"""
import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from project.models.image_classification import get_model
from project.dsets.vision.dataset import CIFAR10CDataset

DATA = '/home/kairisasaki/data/cifar10'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MEAN = (0.4915, 0.4823, 0.4468)
STD  = (0.2470, 0.2435, 0.2616)

TARGETS = [
    dict(name='apr_s_afa_ep250',  path='results/afa_ep250_s0/best.pt'),
    dict(name='apr_s_wca_ep250',  path='results/afa-wca_ep250_s0/best.pt'),
]

normalise   = T.Normalize(mean=MEAN, std=STD)
test_transform = T.Compose([T.ToTensor()])

def eval_loader(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += model(normalise(x)).argmax(1).eq(y).sum().item()
            total   += y.size(0)
    return correct / total

for t in TARGETS:
    model = get_model('c10', 'rn18')(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(t['path'], map_location=DEVICE))
    model.eval()

    severities  = [1, 2, 3, 4, 5]
    corruptions = CIFAR10CDataset.corruptions   # 19 種

    results = {}
    for corruption in corruptions:
        accs = []
        for sev in severities:
            ds  = CIFAR10CDataset(DATA, severity=sev, corruption=corruption,
                                  transform=test_transform)
            dl  = DataLoader(ds, batch_size=512, shuffle=False,
                             num_workers=4, pin_memory=True)
            accs.append(eval_loader(model, dl))
        results[corruption] = sum(accs) / len(accs)
        print(f'  {corruption:<25}: {results[corruption]*100:.2f}%')

    corr_avg = sum(results.values()) / len(results)
    print(f'\n[{t["name"]}]  corr_avg={corr_avg*100:.4f}%\n')
