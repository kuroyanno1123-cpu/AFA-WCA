"""
素の AugMix（JSD loss、adversarial stream なし）を 200ep で実行。

AugMix+AFA / AugMix+WCA との対照実験として位置づける。
  - Model: rn18_dubn (他実験と同じ。DuBN は 'M' route のみ使用)
  - Loss:  JSD (元 AugMix 論文設定)
  - Aug:   AugMix のみ、周波数拡張なし
"""
import os
os.environ['WANDB_MODE'] = 'offline'

from config_utils import ConfigBuilder
from main import main

DATA = '/home/kairisasaki/data/cifar10'

config = ConfigBuilder.build(
    ds='c10', m='rn18_dubn',
    attack='none',          # adversarial stream なし
    use_augmix=True,
    use_jsd=True,           # JSD loss (元 AugMix 設定)
    use_prime=False, use_fourier=False, use_wca=False, use_apr=False,
    in_mix=False, use_mix=False,
    premix='none', min_str=0., mean_str=5.,
)
config.data_dir = DATA
config.num_workers = 4
config.project = 'afa-wca-c10'
config.run_name = 'augmix_plain_c10_200ep'

print(f'use_jsd={config.use_jsd}')
print(f'enable_attack={config.enable_attack}')
print(f'use_augmix={config.enable_aug.use_augmix}')
print(f'epochs={config.epochs}  batch={config.batch_size}')

main(config)
