"""
AugMix+WCA / PRIME+AFA / PRIME+WCA を 200ep で実行（augmix_afa は学習済みのためスキップ）。
"""
import os
os.environ['WANDB_MODE'] = 'offline'

from config_utils import ConfigBuilder
from main import main

DATA = '/home/kairisasaki/data/cifar10'

COMMON = dict(
    ds='c10', m='rn18_dubn',
    in_mix=False, use_jsd=False, use_mix=False,
    use_apr=False, premix='none',
    min_str=0., mean_str=5.,
)

EXPERIMENTS = [
    dict(
        name='augmix_wca',
        attack='wca',   use_augmix=True,  use_prime=False,
        use_fourier=False, use_wca=False,
    ),
    dict(
        name='prime_afa',
        attack='prime', use_augmix=False, use_prime=True,
        use_fourier=True,  use_wca=False,
    ),
    dict(
        name='prime_wca',
        attack='prime', use_augmix=False, use_prime=True,
        use_fourier=False, use_wca=True,
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
    print(f'{"=" * 60}')

    main(config)
