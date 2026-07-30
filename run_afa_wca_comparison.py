"""
AugMix+AFA / AugMix+WCA / PRIME+AFA / PRIME+WCA を公式設定 200ep で実行。

公式経路: main.py → AdvModule → build_augmentations + make_attack → rn18_dubn
WCA は attack.type='wca' または use_wca=True の 1 フラグ切り替えのみ。

実行:
    conda run -n apr python run_afa_wca_comparison.py
"""
import os
os.environ['WANDB_MODE'] = 'offline'

from config_utils import ConfigBuilder
from main import main

DATA = '/home/kairisasaki/data/cifar10'

# ── 公式 C10 共通設定 ─────────────────────────────────────────────────────
COMMON = dict(
    ds='c10', m='rn18_dubn',
    in_mix=False, use_jsd=False, use_mix=False,
    use_apr=False, premix='none',
    min_str=0., mean_str=5.,   # CIFAR-10 AFA 公式 FBA params
)

# ── 4 構成 ────────────────────────────────────────────────────────────────
# AFA ↔ WCA の切り替え = 下線部の 1 フラグのみ(他は完全同一)
#
#   AugMix+AFA: attack='afa'   → A: FBA(x_clean),  M: AugMix(x)
#   AugMix+WCA: attack='wca'   → A: WCA(x_clean),  M: AugMix(x)
#
#   PRIME+AFA:  use_fourier=True  → A: PRIME(x), M: FBA(PRIME(x))
#   PRIME+WCA:  use_wca=True      → A: PRIME(x), M: WCA(PRIME(x))

EXPERIMENTS = [
    dict(
        name='augmix_afa',
        attack='afa',   use_augmix=True,  use_prime=False,
        use_fourier=False, use_wca=False,
    ),
    dict(
        name='augmix_wca',
        attack='wca',   use_augmix=True,  use_prime=False,
        use_fourier=False, use_wca=False,  # WCA は attack 側
    ),
    dict(
        name='prime_afa',
        attack='prime', use_augmix=False, use_prime=True,
        use_fourier=True,  use_wca=False,
    ),
    dict(
        name='prime_wca',
        attack='prime', use_augmix=False, use_prime=True,
        use_fourier=False, use_wca=True,   # WCA は train_aug 側 (PRIME の後)
    ),
]

for exp in EXPERIMENTS:
    name = exp.pop('name')
    config = ConfigBuilder.build(**COMMON, **exp)
    config.data_dir = DATA
    config.num_workers = 4
    config.project = 'afa-wca-c10'
    config.run_name = f'{name}_c10_200ep'

    print(f'\n{"=" * 60}')
    print(f'  Experiment: {name}')
    print(f'  attack={config.attack.type if config.enable_attack else None}')
    print(f'  use_augmix={config.enable_aug.use_augmix}')
    print(f'  use_prime={config.enable_aug.use_prime}')
    print(f'  use_fourier={config.enable_aug.general_fourier}')
    print(f'  use_wca={config.enable_aug.use_wca}')
    print(f'  epochs={config.epochs}  batch={config.batch_size}')
    print(f'{"=" * 60}')

    main(config)
