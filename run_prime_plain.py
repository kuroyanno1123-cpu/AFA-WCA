"""
素のPRIME: 単一ストリームCE、PRIMEのみ。
  - model  : rn18 (論文の rn18prime に対応、DuBN/DuBIN なし)
  - attack : none → enable_attack=False → BaseModule (単一ストリーム)
  - aug    : PRIME のみ (train_aug)
  - loss   : CE
"""
import os
os.environ['WANDB_MODE'] = 'offline'

from config_utils import ConfigBuilder
from main import main

DATA = '/home/kairisasaki/data/cifar10'

config = ConfigBuilder.build(
    ds='c10', m='rn18',
    attack='none', use_augmix=False, use_prime=True,
    use_fourier=False, use_wca=False, use_jsd=False,
    in_mix=False, use_mix=False, use_apr=False, premix='none',
)
config.data_dir    = DATA
config.num_workers = 4
config.project     = 'afa-wca-c10'
config.run_name    = 'prime_plain_c10_200ep'

print('Experiment: 素のPRIME (rn18, attack=none, single-stream CE)')
main(config)
