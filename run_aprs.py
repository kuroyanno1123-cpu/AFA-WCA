"""
APR-S を公式経路 (AdvModule + rn18_dubn) で 200ep 実行。

構成:
  attack='apr' → make_attack が APR(p=1.0) を返す
  M-route: model(norm(x_clean))   [augmentation なし]
  A-route: model(norm(APR(x)))    [DuBN-A]
  loss: ACE = (CE_M + CE_A) / 2
"""
import os
os.environ['WANDB_MODE'] = 'offline'

from config_utils import ConfigBuilder
from main import main

DATA = '/home/kairisasaki/data/cifar10'

config = ConfigBuilder.build(
    ds='c10', m='rn18_dubn',
    attack='apr',
    use_prime=False, use_augmix=False,
    use_fourier=False, use_wca=False,
    use_apr=False,          # train_aug には APR を入れない (attack 側のみ)
    in_mix=False, use_jsd=False, use_mix=False,
    premix='none', min_str=0., mean_str=5.,
)
config.data_dir = DATA
config.num_workers = 4
config.project = 'afa-wca-c10'
config.run_name = 'apr_s_c10_200ep'

print('attack:', config.attack.type)
print('train_aug will be: Identity (no aug in train_aug)')
print('epochs:', config.epochs, '  batch:', config.batch_size)

main(config)
