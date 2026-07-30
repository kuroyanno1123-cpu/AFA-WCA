"""
比較学習スクリプト (v3) — AFA 論文 Table 3 準拠

=== 論文 (Fig.3, Table 3) の正しい構成 ===
  メイン:  x_main = VisualAug(x)     # PRIME または AugMix を元画像にかけた画像
  補助:   x_aux  = FBA(x) or WCA(x) # 元画像に周波数拡張だけをかけた画像
  損失:   ACE = (CE(model(x_main)) + CE(model(x_aux))) / 2
  ※ 視覚拡張と周波数拡張は別ストリーム (in_mix ではない)

=== base_module.py との対応 ===
  AdvModule.training_step_noaugmix (AugMix なし):
    x_main = attack(x_clean) → AT-free では PRIME(x_clean)
    x_aux  = train_aug(x_clean) = FBA(x_clean) → そのまま FBA/WCA(x_clean)
    loss   = ACE
  AdvModule.training_step_augmix (AugMix あり):
    (x1_clean, x2_augmix) = batch  [retain_clean=True]
    x_main = x2_augmix (AT-free; AT版では attack(x1_clean))
    x_aux  = train_aug(x2_augmix) = FBA/WCA(x2_augmix)  ← AFA公式と一致
    loss   = ACE
  → AdvModule の「分離型」が Table 3 に対応。
    NormalJSDModule の in_mix (aug_config に FBA を混ぜる) は Table 3 とは別構成。

=== 2 モード ===
  A) 標準メイン × 補助 (旧来)
       --aug {afa-wca, afa, apr-s}
       x_main = x_clean, x_aux = aug(x_clean)
       出力: results/<aug>_ep<N>_s<seed>/

  B) PRIME/AugMix メイン × FBA/WCA 補助 (新規、論文準拠)
       --main {prime, augmix}  --aux {fba, wca}
       x_main = VisualAug(x),  x_aux = FreqAug(x_clean)  ← 別ストリーム
       出力: results/<main>_<aux>_ep<N>_s<seed>/

共通ハイパラ (AFA 公式 C10):
  ResNet-18, batch=256, SGD lr=0.1 nesterov wd=5e-4
  CosineAnnealingLR per-step (T_max=epochs×steps, eta_min=1e-5)
  grad_clip=1.0, seed=0
  AFA 正規化 (0.4915,0.4823,0.4468)/(0.2470,0.2435,0.2616)
  CIFAR-10-C 15 腐敗×5 深刻度 mCE
"""

import sys, os, argparse, csv, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.datasets import CIFAR10

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── 定数 ─────────────────────────────────────────────────────────────────
DATA   = '/home/kairisasaki/data/cifar10'
DATA_C = '/home/kairisasaki/APR_phase/data'
MEAN   = (0.4915, 0.4823, 0.4468)
STD    = (0.2470, 0.2435, 0.2616)
IMG_SIZE = 32

CORRUPTIONS_15 = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ─── 周波数/補助拡張 ───────────────────────────────────────────────────────
def get_freq_aug(name, args):
    """Tensor [0,1] → Tensor [0,1] の補助拡張を返す"""
    if name in ('fba', 'afa', 'afa-wca'):
        from project.augs.fba.fourier_basis import GeneralFourierOnline
        return GeneralFourierOnline(
            img_size=IMG_SIZE, groups=range(1, IMG_SIZE + 1), phases=(0., 1.),
            f_cut=1, phase_cut=1, min_str=0, mean_str=5, granularity=448,
        )
    elif name == 'wca':
        from project.augs.wca import WaveletBasisSwapOnline
        return WaveletBasisSwapOnline(
            source_wavelet=args.wca_source, target_wavelet=args.wca_target,
            level=args.wca_level, swap_prob=args.wca_swap_prob,
        )
    elif name == 'apr-s':
        from project.augs.apr import APR
        return APR(p=0.6)
    else:
        raise ValueError(f'Unknown freq aug: {name}')


# ─── データセット / 拡張の構築 ─────────────────────────────────────────────
def build_datasets_and_aug(args):
    """
    返り値: (train_dataset, val_dataset, main_aug, freq_aug, batch_mode)

    batch_mode='simple':
      batch = (x, y)
      x_main = x,          x_aux = freq_aug(x)      [旧来モード]

    batch_mode='prime':
      batch = (x, y)
      x_main = main_aug(x) = PRIME(x)
      x_aux  = freq_aug(x)                          [同じ x から別ストリーム]

    batch_mode='augmix':
      batch = ((x_clean, x_augmix), y)
      x_main = x_augmix    = AugMix(x)
      x_aux  = freq_aug(x_augmix)                   [AFA公式: train_aug(x2_augmix)]
      ← AdvModule.training_step_augmix の AT-free 版に対応
    """
    train_tf = T.Compose([
        T.RandomCrop(IMG_SIZE, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
    ])
    val_dataset = CIFAR10(DATA, train=False, transform=T.ToTensor(), download=False)

    if args.main == 'none':
        # ── モード A: 標準メイン × 補助 ──────────────────────────────────
        train_dataset = CIFAR10(DATA, train=True, transform=train_tf, download=False)
        freq_aug = get_freq_aug(args.aug, args)
        return train_dataset, val_dataset, None, freq_aug, 'simple'

    elif args.main == 'prime':
        # ── モード B-1: PRIME メイン × FBA/WCA 補助 (分離型) ─────────────
        # AdvModule.training_step_noaugmix の AT-free 版:
        #   x_main = PRIME(x),  x_aux = FBA(x)  — 同じ x から独立
        from project.augs.prime import (
            GeneralizedPRIMEModule, PRIMEAugModule, make_original_prime_aug_config,
        )
        train_dataset = CIFAR10(DATA, train=True, transform=train_tf, download=False)
        aug_config = make_original_prime_aug_config('c10')  # FBA/WCA は混ぜない
        prime = GeneralizedPRIMEModule(
            mixture_width=3, mixture_depth=-1, max_depth=3,
            aug_module=PRIMEAugModule(aug_config),
        )
        freq_aug = get_freq_aug(args.aux, args)
        return train_dataset, val_dataset, prime, freq_aug, 'prime'

    elif args.main == 'augmix':
        # ── モード B-2: AugMix メイン × FBA/WCA 補助 (分離型) ────────────
        # AdvModule.training_step_augmix の AT-free 版:
        #   (x_clean, x_augmix) = batch
        #   x_main = x_augmix,  x_aux = FBA(x_clean)  — 別ストリーム
        from project.augs import AugMixDataset
        pil_dataset = CIFAR10(DATA, train=True, transform=None, download=False)
        train_dataset = AugMixDataset(
            pil_dataset,
            all_ops=True,
            extra_ops=[],          # FBA/WCA は AugMix に混ぜない (分離型)
            preprocess=train_tf,
            aug_severity=3,
            max_depth=3,
            mixture_width=3,
            mixture_depth=-1,
            no_jsd=True,
            retain_clean=True,     # ((x_clean, x_augmix), y) を返す
            img_sz=IMG_SIZE,
        )
        freq_aug = get_freq_aug(args.aux, args)
        return train_dataset, val_dataset, None, freq_aug, 'augmix'

    else:
        raise ValueError(f'Unknown main: {args.main}')


# ─── CIFAR-10-C 評価 ──────────────────────────────────────────────────────
def eval_cifar10c(model, normalise, device, batch_size=512):
    c10c_dir = os.path.join(DATA_C, 'CIFAR-10-C')
    labels   = np.load(os.path.join(c10c_dir, 'labels.npy'))
    model.eval()
    results = {}
    with torch.no_grad():
        for corruption in CORRUPTIONS_15:
            data = np.load(os.path.join(c10c_dir, f'{corruption}.npy'))
            accs = []
            for sev in range(5):
                imgs = data[sev*10000:(sev+1)*10000]
                labs = labels[sev*10000:(sev+1)*10000]
                correct = total = 0
                for i in range(0, len(imgs), batch_size):
                    xb = torch.from_numpy(imgs[i:i+batch_size]).float().permute(0,3,1,2).div(255.)
                    yb = torch.from_numpy(labs[i:i+batch_size]).long().to(device)
                    out = model(normalise(xb.to(device)))
                    correct += out.argmax(1).eq(yb).sum().item()
                    total   += yb.size(0)
                accs.append(correct / total * 100.)
            results[corruption] = float(np.mean(accs))
    return float(np.mean(list(results.values()))), results


def eval_clean(model, loader, normalise, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += model(normalise(x)).argmax(1).eq(y).sum().item()
            total   += y.size(0)
    return correct / total * 100.


# ─── 1 エポック学習 ────────────────────────────────────────────────────────
def train_epoch(model, optimizer, scheduler, loader,
                main_aug, freq_aug, normalise, device, grad_clip, batch_mode):
    """
    batch_mode='simple' : batch=(x,y) → x_main=x, x_aux=freq_aug(x)
    batch_mode='prime'  : batch=(x,y) → x_main=main_aug(x), x_aux=freq_aug(x_main)
    batch_mode='augmix' : batch=((x_clean,x_aug),y) → x_main=x_aug, x_aux=freq_aug(x_aug)
    """
    model.train()
    sum_loss = sum_cm = sum_ca = n = 0

    for batch in loader:
        if batch_mode == 'augmix':
            (x_clean, x_aug), y = batch
            x_clean = x_clean.to(device)
            x_aug   = x_aug.to(device)
            y       = y.to(device)
            x_main = x_aug               # AugMix(x) — 視覚拡張ストリーム
            x_aux  = freq_aug(x_aug)     # FBA/WCA(AugMix(x)) — AFA公式: train_aug(x2_augmix)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            if batch_mode == 'prime':
                x_main = main_aug(x)         # PRIME(x) — 視覚拡張ストリーム
                x_aux  = freq_aug(x_main)    # FBA/WCA(PRIME(x)) — AFA公式: train_aug=T.Compose([PRIME,FBA])
            else:  # 'simple'
                x_main = x                   # 素の画像 (旧来)
                x_aux  = freq_aug(x)         # FBA/WCA/APR(x)

        logits_m = model(normalise(x_main))
        logits_a = model(normalise(x_aux))
        ce_m = F.cross_entropy(logits_m, y)
        ce_a = F.cross_entropy(logits_a, y)
        loss = (ce_m + ce_a) / 2.

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()   # AFA 公式: バッチ単位でステップ

        sum_loss += loss.item(); sum_cm += ce_m.item(); sum_ca += ce_a.item()
        n += 1

    return sum_loss / n, sum_cm / n, sum_ca / n


# ─── main ─────────────────────────────────────────────────────────────────
def main(args):
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if args.main == 'none':
        tag = f'{args.aug}_ep{args.max_epoch}_s{args.seed}'
    else:
        tag = f'{args.main}_{args.aux}_ep{args.max_epoch}_s{args.seed}'
    outdir = os.path.join('./results', tag)
    os.makedirs(outdir, exist_ok=True)
    print(f'[config] tag={tag}  device={device}')

    from project.models.image_classification import get_model
    model = get_model('c10', 'rn18')(num_classes=10).to(device)
    print(f'[model]  params={sum(p.numel() for p in model.parameters()):,}')

    normalise = T.Normalize(mean=MEAN, std=STD)

    train_dataset, val_dataset, main_aug, freq_aug, batch_mode = build_datasets_and_aug(args)
    t_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                      num_workers=4, pin_memory=True)
    v_dl = DataLoader(val_dataset,   batch_size=512, shuffle=False,
                      num_workers=4, pin_memory=True)
    steps_per_epoch = len(t_dl)  # 実バッチ数 (端数バッチ含む; AFA公式 C10=196 に対応)
    print(f'[aug]    batch_mode={batch_mode}  main={main_aug}  freq={freq_aug}')
    print(f'[sched]  steps_per_epoch={steps_per_epoch}  T_max={args.max_epoch * steps_per_epoch}')

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=5e-4, nesterov=True,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epoch * steps_per_epoch,
        eta_min=1e-5,
    )

    log_path = os.path.join(outdir, 'logs.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','ace','ce_main','ce_aux','clean_acc','lr','time_min'])

    best_acc = 0.
    for epoch in range(1, args.max_epoch + 1):
        t0 = time.time()
        ace, ce_m, ce_a = train_epoch(
            model, optimizer, scheduler, t_dl,
            main_aug, freq_aug, normalise, device, args.grad_clip, batch_mode,
        )
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

    print('\n==> CIFAR-10-C eval (best checkpoint)...')
    model.load_state_dict(torch.load(os.path.join(outdir, 'best.pt'), map_location=device))
    mce, per_c = eval_cifar10c(model, T.Normalize(mean=MEAN, std=STD), device)

    result_path = os.path.join(outdir, 'cifar10c_results.txt')
    with open(result_path, 'w') as f:
        f.write(f'tag={tag}\n')
        f.write(f'best_clean_acc={best_acc:.2f}\n')
        f.write(f'mCE_15={mce:.4f}\n\n')
        for c, v in sorted(per_c.items()):
            f.write(f'  {c:<25s}: {v:.4f}\n')
    print(f'\n[result] {tag}  clean={best_acc:.2f}%  mCE={mce:.4f}%')
    print(f'[saved]  {result_path}')


# ─── CLI ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--aug',  choices=['afa-wca', 'afa', 'apr-s'], default=None,
                   help='モード A: 補助拡張 (--main none 時)')
    p.add_argument('--main', choices=['none', 'prime', 'augmix'], default='none')
    p.add_argument('--aux',  choices=['fba', 'wca'], default=None,
                   help='モード B: 周波数補助 (--main prime/augmix 時)')
    p.add_argument('--max-epoch',    type=int,   default=250)
    p.add_argument('--batch-size',   type=int,   default=256)
    p.add_argument('--lr',           type=float, default=0.1)
    p.add_argument('--grad-clip',    type=float, default=1.0)
    p.add_argument('--seed',         type=int,   default=0)
    p.add_argument('--gpu',          type=int,   default=0)
    p.add_argument('--wca-source',    default='haar')
    p.add_argument('--wca-target',    default='db8')
    p.add_argument('--wca-level',     type=int,   default=1)
    p.add_argument('--wca-swap-prob', type=float, default=0.2)
    args = p.parse_args()

    if args.main == 'none':
        if args.aug is None:
            p.error('--main none のときは --aug が必要')
    else:
        if args.aux is None:
            p.error('--main prime/augmix のときは --aux が必要')
        args.aug = None
    main(args)
