"""
PRIME+WCA-LL: WCAのLL帯も基底交換あり
PRIME+WCA (swap_ll=False) との比較用
M-route = WCA_LL(PRIME_1(x)), A-route = PRIME_2(x)
"""
import os
os.environ['WANDB_MODE'] = 'offline'

from config_utils import ConfigBuilder
from main import main

config = ConfigBuilder.build(
    ds='c10', m='rn18_dubn',
    attack='prime',
    in_mix=False, use_mix=False, use_apr=False, premix='none',
    use_augmix=False, use_jsd=False,
    use_prime=True, use_fourier=False,
    use_wca=True, wca_swap_ll=True,
    min_str=0., mean_str=5.,
)
config.data_dir    = '/home/kairisasaki/data/cifar10'
config.num_workers = 4
config.project     = 'afa-wca-c10'
config.run_name    = 'prime_wca_ll_c10_200ep'
config.epochs      = 200

print('PRIME+WCA-LL (swap_ll=True, 200ep)')
main(config)
