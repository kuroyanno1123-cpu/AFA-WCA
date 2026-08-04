"""
素のPRIME: PRIMEアタックのみ、AFA/WCAなし。
"""
import os
os.environ['WANDB_MODE'] = 'offline'

from config_utils import ConfigBuilder
from main import main

DATA = '/home/kairisasaki/data/cifar10'

config = ConfigBuilder.build(
    ds='c10', m='rn18_dubn',
    attack='prime', use_augmix=False, use_prime=True,
    use_fourier=False, use_wca=False, use_jsd=False,
    in_mix=False, use_mix=False, use_apr=False, premix='none',
    min_str=0., mean_str=5.,
)
config.data_dir    = DATA
config.num_workers = 4
config.project     = 'afa-wca-c10'
config.run_name    = 'prime_plain_c10_200ep'

print('Experiment: 素のPRIME (PRIME attack only, no AFA/WCA)')
main(config)
