"""
AFA-WCA / AFA / APR-S  比較学習スクリプト
全3手法を AFA-WCA リポジトリで統一実行。

補助ストリームの差し替えのみで他は完全同一:
  aug=afa-wca : WaveletBasisSwapOnline (今回の実装)
  aug=afa     : GeneralFourierOnline   (AFA公式 FBA)
  aug=apr-s   : APR (FFT振幅スワップ, AFA公式実装)

共通ハイパラ (AFA公式 C10 設定):
  model       : ResNet18 (project/models/image_classification/cifar.py)
  normalise   : AFA CIFAR-10 mean/std
  batch_size  : 256
  optimizer   : SGD lr=0.1, momentum=0.9, wd=5e-4, nesterov=True
  scheduler   : CosineAnnealingLR (step-wise, eta_min=1e-5)
  grad_clip   : 1.0
  loss        : ACE = (CE_main + CE_aux) / 2
  eval        : CIFAR-10-C 15腐敗 × 5深刻度 mCE
"""
import sys, os, argparse, csv, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── データ/評価設定 ──────────────────────────────────────────────────────
DATA   = '/home/kairisasaki/data/cifar10'
DATA_C = '/home/kairisasaki/APR_phase/data'

# AFA 公式 CIFAR-10 正規化 (project/dsets/vision/dataset.py)
MEAN = (0.4915, 0.4823, 0.4468)
STD  = (0.2470, 0.2435, 0.2616)

# WCA / AFA 共通 15腐敗
CORRUPTIONS_15 = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_aug(aug_name, args):
    if aug_name == 'afa-wca':
        from project.augs.wca import WaveletBasisSwapOnline
        return WaveletBasisSwapOnline(
            source_wavelet=args.wca_source, target_wavelet=args.wca_target,
            level=args.wca_level, swap_prob=args.wca_swap_prob,
        )
    elif aug_name == 'afa':
        from project.augs.fba.fourier_basis import GeneralFourierOnline
        return GeneralFourierOnline(
            img_size=32, groups=range(1, 33), phases=(0., 1.),
            f_cut=1, phase_cut=1, min_str=0, mean_str=5, granularity=448,
        )
    elif aug_name == 'apr-s':
        from project.augs.apr import APR
        return APR(p=0.6)
    else:
        raise ValueError(f'Unknown aug: {aug_name}')


# ─── CIFAR-10-C 評価 ──────────────────────────────────────────────────────
def eval_cifar10c(model, normalise, device, data_c_dir, batch_size):
    c10c_dir = os.path.join(data_c_dir, 'CIFAR-10-C')
    labels   = np.load(os.path.join(c10c_dir, 'labels.npy'))
    model.eval()
    results = {}
    with torch.no_grad():
        for corruption in CORRUPTIONS_15:
            data = np.load(os.path.join(c10c_dir, f'{corruption}.npy'))
            accs = []
            for sev in range(5):
                imgs = data[sev * 10000:(sev + 1) * 10000]
                labs = labels[sev * 10000:(sev + 1) * 10000]
                correct = total = 0
                for i in range(0, len(imgs), batch_size):
                    xb = torch.from_numpy(imgs[i:i+batch_size]).float().permute(0,3,1,2).div(255.)
                    yb = torch.from_numpy(labs[i:i+batch_size]).long().to(device)
                    out = model(normalise(xb.to(device)))
                    correct += out.argmax(1).eq(yb).sum().item()
                    total   += yb.size(0)
                accs.append(correct / total * 100.)
            results[corruption] = np.mean(accs)
    return np.mean(list(results.values())), results


# ─── 1エポック学習 ────────────────────────────────────────────────────────
def train_epoch(model, optimizer, scheduler, loader, aug, normalise, device, grad_clip):
    model.train()
    sum_ce_main = sum_ce_aux = sum_loss = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_aux = aug(x)
        logits_m = model(normalise(x))
        logits_a = model(normalise(x_aux))
        ce_m = F.cross_entropy(logits_m, y)
        ce_a = F.cross_entropy(logits_a, y)
        loss = (ce_m + ce_a) / 2.
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()  # AFA公式: バッチ単位でステップ
        sum_ce_main += ce_m.item(); sum_ce_aux += ce_a.item()
        sum_loss += loss.item(); n += 1
    return sum_loss/n, sum_ce_main/n, sum_ce_aux/n


def eval_clean(model, loader, normalise, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += model(normalise(x)).argmax(1).eq(y).sum().item()
            total   += y.size(0)
    return correct / total * 100.


# ─── main ─────────────────────────────────────────────────────────────────
def main(args):
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'[config] aug={args.aug}  epochs={args.max_epoch}  seed={args.seed}  device={device}')

    tag    = f'{args.aug}_ep{args.max_epoch}_s{args.seed}'
    outdir = os.path.join('./results', tag)
    os.makedirs(outdir, exist_ok=True)

    # モデル (AFA公式 ResNet18)
    from project.models.image_classification import get_model
    model = get_model('c10', 'rn18')(num_classes=10).to(device)
    print(f'[model]  params={sum(p.numel() for p in model.parameters()):,}')

    normalise = T.Normalize(mean=MEAN, std=STD)

    # データ (AFA公式 標準前処理)
    from torchvision.datasets import CIFAR10
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    test_tf  = T.Compose([T.ToTensor()])
    t_dset   = CIFAR10(DATA, train=True,  transform=train_tf, download=False)
    v_dset   = CIFAR10(DATA, train=False, transform=test_tf,  download=False)
    steps_per_epoch = len(t_dset) // args.batch_size
    t_dl = DataLoader(t_dset, batch_size=args.batch_size, shuffle=True,
                      num_workers=4, pin_memory=True)
    v_dl = DataLoader(v_dset, batch_size=512, shuffle=False,
                      num_workers=4, pin_memory=True)

    # 拡張
    aug = get_aug(args.aug, args)
    print(f'[aug]    {aug}')

    # オプティマイザ / スケジューラ (AFA公式 C10)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=5e-4, nesterov=True,
    )
    # step-wise CosineAnnealingLR (AFA公式と同一)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epoch * steps_per_epoch,
        eta_min=1e-5,
    )

    # CSVログ
    log_path = os.path.join(outdir, 'logs.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','ace','ce_main','ce_aux','clean_acc','lr','time_min'])

    best_acc = 0.
    for epoch in range(1, args.max_epoch + 1):
        t0 = time.time()
        ace, ce_m, ce_a = train_epoch(
            model, optimizer, scheduler, t_dl, aug, normalise, device, args.grad_clip)

        clean_acc = eval_clean(model, v_dl, normalise, device)
        if clean_acc > best_acc:
            best_acc = clean_acc
            torch.save(model.state_dict(), os.path.join(outdir, 'best.pt'))

        elapsed = (time.time() - t0) / 60.
        lr_now  = scheduler.get_last_lr()[0]

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{ace:.4f}', f'{ce_m:.4f}', f'{ce_a:.4f}',
                                     f'{clean_acc:.2f}', f'{lr_now:.6f}', f'{elapsed:.1f}'])

        if epoch % 10 == 0 or epoch == 1:
            print(f'Ep {epoch:3d}/{args.max_epoch}  '
                  f'ACE={ace:.4f}  CE_main={ce_m:.4f}  CE_aux={ce_a:.4f}  '
                  f'clean={clean_acc:.2f}%  lr={lr_now:.5f}  ({elapsed:.1f}min)')

    # ── CIFAR-10-C 評価 ──────────────────────────────────────────────────
    print('\n==> CIFAR-10-C eval (best checkpoint)...')
    model.load_state_dict(torch.load(os.path.join(outdir, 'best.pt')))
    mce, per_c = eval_cifar10c(model, T.Normalize(mean=MEAN, std=STD), device, DATA_C, 512)

    result_path = os.path.join(outdir, 'cifar10c_results.txt')
    with open(result_path, 'w') as f:
        f.write(f'aug={args.aug}  epochs={args.max_epoch}  seed={args.seed}\n')
        f.write(f'best_clean_acc={best_acc:.2f}\n')
        f.write(f'mCE_15={mce:.4f}\n\n')
        for c, v in sorted(per_c.items()):
            f.write(f'  {c:<25s}: {v:.4f}\n')

    print(f'\n[result] {args.aug}  clean={best_acc:.2f}%  mCE={mce:.4f}%')
    print(f'[saved]  {result_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--aug',          choices=['afa-wca', 'afa', 'apr-s'], required=True)
    p.add_argument('--max-epoch',    type=int,   default=200)
    p.add_argument('--batch-size',   type=int,   default=256)
    p.add_argument('--lr',           type=float, default=0.1)
    p.add_argument('--grad-clip',    type=float, default=1.0)
    p.add_argument('--seed',         type=int,   default=0)
    p.add_argument('--gpu',          type=int,   default=0)
    # AFA-WCA ハイパラ
    p.add_argument('--wca-source',    default='haar')
    p.add_argument('--wca-target',    default='db8')
    p.add_argument('--wca-level',     type=int,   default=1)
    p.add_argument('--wca-swap-prob', type=float, default=0.2)
    args = p.parse_args()
    main(args)
