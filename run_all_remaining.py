"""
AugMix+WCA → PRIME+AFA → 素のAugMix(JSD) → PRIME+WCA の順で実行。
"""
import os
os.environ['WANDB_MODE'] = 'offline'

from config_utils import ConfigBuilder
from main import main

DATA = '/home/kairisasaki/data/cifar10'

COMMON = dict(
    ds='c10', m='rn18_dubn',
    in_mix=False, use_mix=False,
    use_apr=False, premix='none',
    min_str=0., mean_str=5.,
)

EXPERIMENTS = [
    dict(
        name='augmix_wca',
        attack='wca',   use_augmix=True,  use_prime=False,
        use_fourier=False, use_wca=False, use_jsd=False,
    ),
    dict(
        name='augmix_plain',
        attack='none',  use_augmix=True,  use_prime=False,
        use_fourier=False, use_wca=False, use_jsd=True,
    ),
    dict(
        name='prime_afa',
        attack='prime', use_augmix=False, use_prime=True,
        use_fourier=True,  use_wca=False, use_jsd=False,
    ),
    dict(
        name='prime_wca',
        attack='prime', use_augmix=False, use_prime=True,
        use_fourier=False, use_wca=True,  use_jsd=False,
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
    print(f'{"=" * 60}')

    main(config)
