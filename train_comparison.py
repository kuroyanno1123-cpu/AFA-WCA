"""
比較学習スクリプト (v2)

2 モード:
  A) 標準メイン × 補助ストリーム(旧来)
       --aug {afa-wca, afa, apr-s}
       出力: results/<aug>_ep<N>_s<seed>/

  B) PRIME/AugMix メイン × FBA/WCA 補助(新規)
       --main {prime, augmix}  --aux {fba, wca}
       出力: results/<main>_<aux>_ep<N>_s<seed>/

共通ハイパラ (AFA 公式 C10):
  ResNet-18, batch=256, SGD lr=0.1 nesterov wd=5e-4
  CosineAnnealingLR per-step (T_max=epochs×steps, eta_min=1e-5)
  grad_clip=1.0, seed=0, ACE=(CE_main+CE_aux)/2
  AFA 正規化 (0.4915,0.4823,0.4468)/(0.2470,0.2435,0.2616)
  CIFAR-10-C 15 腐敗×5 深刻度 mCE

PRIME + FBA/WCA:
  FBA/WCA を PRIMEAugModule の aug_config に append (AFA 公式 in_mix と同一経路)
  PRIMEAugModule は各 op を (B,C,H,W) float [0,1] で呼び出す ← 両方 OK

AugMix + FBA/WCA:
  AugMixDataset の extra_ops に PIL→PIL ラッパー経由で追加 (AFA 公式 in_mix と同一経路)
  AFA 公式は T.PILToTensor()(uint8) を使うが float 前提の aug と非整合なため
  T.ToTensor() (float [0,1]) を使う最小アダプタを噛ませる
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

CORRUPTIONS_15 = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]

IMG_SIZE = 32  # CIFAR-10


# ─── シード ────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ─── 補助拡張の生成 ────────────────────────────────────────────────────────
def get_aux_aug(name, args):
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
        raise ValueError(f'Unknown aux aug: {name}')


def make_pil_aug_op(tensor_aug):
    """Tensor[0,1]→Tensor[0,1] の aug を AugMix 用 PIL→PIL ラッパーに変換"""
    to_tensor = T.ToTensor()     # PIL (H,W,C uint8) → float32 (C,H,W) [0,1]
    to_pil    = T.ToPILImage()   # float32 (C,H,W) [0,1] → PIL
    def op(pil_img, severity=None):
        x = to_tensor(pil_img)
        x = tensor_aug(x)
        return to_pil(x.clamp(0., 1.))
    return op


# ─── データセット / 拡張の構築 ─────────────────────────────────────────────
def build_datasets_and_aug(args):
    """
    返り値: (train_dataset, val_dataset, train_aug, batch_mode)
      train_aug:  None (AugMix 時は dataset に組み込み済み) or Module
      batch_mode: 'simple'  → batch = (x, y)
                  'augmix'  → batch = ((x_clean, x_aug), y)
    """
    train_tf = T.Compose([
        T.RandomCrop(IMG_SIZE, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
    ])
    test_tf = T.ToTensor()
    val_dataset = CIFAR10(DATA, train=False, transform=test_tf, download=False)

    if args.main == 'none':
        # ── モード A: 標準メイン × 補助ストリーム ──────────────────────────
        train_dataset = CIFAR10(DATA, train=True, transform=train_tf, download=False)
        aug = get_aux_aug(args.aug, args)
        return train_dataset, val_dataset, aug, 'simple'

    elif args.main == 'prime':
        # ── モード B-1: PRIME + FBA/WCA ────────────────────────────────────
        from project.augs.prime import (
            GeneralizedPRIMEModule, PRIMEAugModule, make_original_prime_aug_config,
        )
        train_dataset = CIFAR10(DATA, train=True, transform=train_tf, download=False)
        aug_config = make_original_prime_aug_config('c10')
        aux = get_aux_aug(args.aux, args)   # FBA or WCA
        aug_config.append(aux)              # AFA 公式 in_mix と同一経路
        prime = GeneralizedPRIMEModule(
            mixture_width=3, mixture_depth=-1, max_depth=3,
            aug_module=PRIMEAugModule(aug_config),
        )
        return train_dataset, val_dataset, prime, 'simple'

    elif args.main == 'augmix':
        # ── モード B-2: AugMix + FBA/WCA ───────────────────────────────────
        from project.augs import AugMixDataset
        # AugMix は PIL 入力が必要 → transform=None で PIL を返す
        pil_dataset = CIFAR10(DATA, train=True, transform=None, download=False)
        aux = get_aux_aug(args.aux, args)
        extra_ops = [make_pil_aug_op(aux)]   # AFA 公式 in_mix と同一経路
        train_dataset = AugMixDataset(
            pil_dataset,
            all_ops=True,
            extra_ops=extra_ops,
            preprocess=train_tf,
            aug_severity=3,
            max_depth=3,
            mixture_width=3,
            mixture_depth=-1,
            no_jsd=True,      # JSD は使わない(ACE loss のみ)
            retain_clean=True, # (x_clean, x_aug) のペアを返す
            img_sz=IMG_SIZE,
        )
        return train_dataset, val_dataset, None, 'augmix'

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
            data  = np.load(os.path.join(c10c_dir, f'{corruption}.npy'))
            accs  = []
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
def train_epoch(model, optimizer, scheduler, loader, aug, normalise, device,
                grad_clip, batch_mode='simple'):
    model.train()
    sum_loss = sum_cm = sum_ca = n = 0
    for batch in loader:
        if batch_mode == 'augmix':
            # AugMixDataset(retain_clean=True) → ((x_clean, x_aug), y)
            (x, x_aux), y = batch
            x, x_aux, y = x.to(device), x_aux.to(device), y.to(device)
        else:
            # 標準 / PRIME → (x, y)、補助は aug で生成
            x, y = batch
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
        scheduler.step()    # AFA 公式: バッチ単位でステップ

        sum_loss += loss.item(); sum_cm += ce_m.item(); sum_ca += ce_a.item()
        n += 1

    return sum_loss / n, sum_cm / n, sum_ca / n


# ─── main ─────────────────────────────────────────────────────────────────
def main(args):
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 出力ディレクトリ
    if args.main == 'none':
        tag = f'{args.aug}_ep{args.max_epoch}_s{args.seed}'
    else:
        tag = f'{args.main}_{args.aux}_ep{args.max_epoch}_s{args.seed}'
    outdir = os.path.join('./results', tag)
    os.makedirs(outdir, exist_ok=True)

    print(f'[config] tag={tag}  device={device}')

    # モデル
    from project.models.image_classification import get_model
    model = get_model('c10', 'rn18')(num_classes=10).to(device)
    print(f'[model]  params={sum(p.numel() for p in model.parameters()):,}')

    normalise = T.Normalize(mean=MEAN, std=STD)

    # データセット / 拡張
    train_dataset, val_dataset, aug, batch_mode = build_datasets_and_aug(args)
    steps_per_epoch = len(train_dataset) // args.batch_size
    t_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                      num_workers=4, pin_memory=True)
    v_dl = DataLoader(val_dataset,   batch_size=512, shuffle=False,
                      num_workers=4, pin_memory=True)

    print(f'[aug]    mode={batch_mode}  aug={aug}')

    # オプティマイザ / スケジューラ
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=5e-4, nesterov=True,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epoch * steps_per_epoch,
        eta_min=1e-5,
    )

    # CSV ログ
    log_path = os.path.join(outdir, 'logs.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','ace','ce_main','ce_aux','clean_acc','lr','time_min'])

    best_acc = 0.
    for epoch in range(1, args.max_epoch + 1):
        t0 = time.time()
        ace, ce_m, ce_a = train_epoch(
            model, optimizer, scheduler, t_dl, aug, normalise, device,
            args.grad_clip, batch_mode=batch_mode,
        )
        clean_acc = eval_clean(model, v_dl, normalise, device)
        if clean_acc > best_acc:
            best_acc = clean_acc
            torch.save(model.state_dict(), os.path.join(outdir, 'best.pt'))

        elapsed  = (time.time() - t0) / 60.
        lr_now   = scheduler.get_last_lr()[0]

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{ace:.4f}', f'{ce_m:.4f}', f'{ce_a:.4f}',
                                    f'{clean_acc:.2f}', f'{lr_now:.6f}', f'{elapsed:.1f}'])

        if epoch % 10 == 0 or epoch == 1:
            print(f'Ep {epoch:3d}/{args.max_epoch}  '
                  f'ACE={ace:.4f}  CE_main={ce_m:.4f}  CE_aux={ce_a:.4f}  '
                  f'clean={clean_acc:.2f}%  lr={lr_now:.5f}  ({elapsed:.1f}min)')

    # CIFAR-10-C 評価
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

    # モード A: 旧来の標準メイン × 補助ストリーム
    p.add_argument('--aug', choices=['afa-wca', 'afa', 'apr-s'], default=None,
                   help='モード A: 補助拡張 (--main は none)')

    # モード B: PRIME/AugMix メイン × FBA/WCA 補助
    p.add_argument('--main', choices=['none', 'prime', 'augmix'], default='none',
                   help='メイン拡張 (none=標準)')
    p.add_argument('--aux', choices=['fba', 'wca'], default=None,
                   help='補助拡張 (--main が prime/augmix の場合)')

    # 共通ハイパラ
    p.add_argument('--max-epoch',    type=int,   default=250)
    p.add_argument('--batch-size',   type=int,   default=256)
    p.add_argument('--lr',           type=float, default=0.1)
    p.add_argument('--grad-clip',    type=float, default=1.0)
    p.add_argument('--seed',         type=int,   default=0)
    p.add_argument('--gpu',          type=int,   default=0)

    # WCA ハイパラ
    p.add_argument('--wca-source',    default='haar')
    p.add_argument('--wca-target',    default='db8')
    p.add_argument('--wca-level',     type=int,   default=1)
    p.add_argument('--wca-swap-prob', type=float, default=0.2)

    args = p.parse_args()

    # バリデーション
    if args.main == 'none':
        if args.aug is None:
            p.error('--main none のときは --aug が必要')
    else:
        if args.aux is None:
            p.error('--main prime/augmix のときは --aux が必要')
        args.aug = None  # モード B では aug フラグは使わない

    main(args)
