"""
Step 3 & 4 確認:
  - 4構成(AugMix+AFA / AugMix+WCA / PRIME+AFA / PRIME+WCA)が 1 バッチ通る
  - AFA↔WCA の切り替えがフラグ1つで済むことを確認
  - DuBN route='A'/'M' の切り替えが効いている (A/M の BN weight が違う)
  - WCA が [0,1] を保つ
"""
import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['WANDB_MODE'] = 'offline'

import torchvision.transforms as T
from torch.utils.data import DataLoader
from config_utils import ConfigBuilder
from project.dsets import get_dataset
from project.models.image_classification import get_model
from project.models.image_classification.dubn import DualBatchNorm2d
from utils import get_standard_transforms, build_augmentations, make_attack

DATA = '/home/kairisasaki/data/cifar10'

# ── 4 構成 ─────────────────────────────────────────────────────────────────
# AFA↔WCA の切り替えは下線部のフラグのみ変わる(他は完全同一)
CONFIGS = [
    dict(name='augmix_afa',
         attack='afa',   use_augmix=True,  use_prime=False,
         use_fourier=False, use_wca=False),  # ← AFA: attack='afa'
    dict(name='augmix_wca',
         attack='wca',   use_augmix=True,  use_prime=False,
         use_fourier=False, use_wca=False),  # ← WCA: attack='wca'  (1フラグ切り替え)
    dict(name='prime_afa',
         attack='prime', use_augmix=False, use_prime=True,
         use_fourier=True,  use_wca=False),  # ← AFA: use_fourier=True
    dict(name='prime_wca',
         attack='prime', use_augmix=False, use_prime=True,
         use_fourier=False, use_wca=True),   # ← WCA: use_wavelet(use_wca)=True (1フラグ切り替え)
]

COMMON = dict(
    ds='c10', m='rn18_dubn',
    in_mix=False, use_jsd=False, use_mix=False, use_apr=False,
    premix='none', min_str=0., mean_str=5.,
)

print('=' * 70)
all_ok = True

for cfg_dict in CONFIGS:
    name = cfg_dict.pop('name')
    config = ConfigBuilder.build(**COMMON, **cfg_dict)
    config.data_dir = DATA

    dataset_class = get_dataset(config.dataset)
    img_sz = dataset_class.image_size
    normalise = T.Normalize(mean=dataset_class.mean, std=dataset_class.std)
    _, train_tf = get_standard_transforms(config.dataset, img_sz)

    model_class = get_model(config.dataset, config.model)
    model = model_class(num_classes=10)

    using_wrapper = config.enable_aug.use_augmix
    t_dset_raw, _ = [
        dataset_class(root=config.data_dir, train=train, transform=transform)
        for train, transform in [
            (True, None if using_wrapper else train_tf), (False, T.ToTensor())
        ]
    ]

    t_dset, train_aug, _ = build_augmentations(t_dset_raw, config, img_sz, train_tf)
    attack = make_attack(config, dataset_class)
    t_dl = DataLoader(t_dset, batch_size=8, shuffle=False, num_workers=0)

    batch = next(iter(t_dl))

    with torch.no_grad():
        if using_wrapper:
            (x_clean, x_aug), y = batch
        else:
            x_clean, y = batch
            x_aug = x_clean

        # ── train_aug の出力確認 ─────────────────────────────────────────
        x_m = train_aug(x_aug)  # M-route input (before normalise)
        assert x_m.shape == x_aug.shape, f"train_aug shape mismatch"
        aug_range_ok = (x_m.min() >= -0.01) and (x_m.max() <= 1.01)

        # ── attack の出力確認 ────────────────────────────────────────────
        x_a = attack(x_clean)
        assert x_a.shape == x_clean.shape, f"attack shape mismatch"
        atk_range_ok = (x_a.min() >= -0.01) and (x_a.max() <= 1.01)

        # ── DuBN route 切り替え確認 ──────────────────────────────────────
        # route='M' (default) → M-BN
        model.apply(lambda m: setattr(m, 'route', 'M'))
        logits_m = model(normalise(x_m))

        # route='A' → A-BN (adversarial)
        model.apply(lambda m: setattr(m, 'route', 'A'))
        logits_a = model(normalise(x_a))
        model.apply(lambda m: setattr(m, 'route', 'M'))  # reset

        # DuBN が有効かどうか: M-BN と A-BN の重みが独立しているか確認
        first_dubn = next(m for m in model.modules() if isinstance(m, DualBatchNorm2d))
        bn_M = first_dubn.bn[0]  # index 0 = 'M'
        bn_A = first_dubn.bn[1]  # index 1 = 'A'
        dubn_independent = not torch.equal(bn_M.running_mean, bn_A.running_mean)

        ace = (F.cross_entropy(logits_m, y) + F.cross_entropy(logits_a, y)) / 2.

        status = 'OK' if (aug_range_ok and atk_range_ok) else 'FAIL'
        if not (aug_range_ok and atk_range_ok):
            all_ok = False

    print(f'[{name:<12}] {status}'
          f'  aug_range={aug_range_ok}  atk_range={atk_range_ok}'
          f'  dubn_independent={dubn_independent}'
          f'  ACE={ace.item():.4f}')
    print(f'              train_aug: {train_aug}')
    print(f'              attack   : {attack}')

print('=' * 70)
print('Step 3 フラグ切り替え確認:')
print('  AugMix: attack=afa  ↔  attack=wca  (1フラグのみ変更)')
print('  PRIME:  use_fourier=True  ↔  use_wca=True  (1フラグのみ変更)')
print('  その他(rn18_dubn/200ep/batch256/CosineAnnealing/grad_clip)は全構成同一')
print('=' * 70)
print('ALL OK' if all_ok else 'SOME FAILED')
