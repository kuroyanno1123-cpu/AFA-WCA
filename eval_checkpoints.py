"""
学習済みチェックポイントから cc_test のみ実行（再学習なし）。

対象:
  1. augmix_afa_c10_200ep  (offline-run-20260730_130508-oayl9tc0)
  2. apr_s_c10_200ep       (offline-run-20260730_161322-alx1wa9d)
"""
import os
os.environ['WANDB_MODE'] = 'offline'

from config_utils import ConfigBuilder
from main import eval_only_main

DATA = '/home/kairisasaki/data/cifar10'
WANDB_DIR = '/home/kairisasaki/AFA-WCA/wandb'

EVAL_TARGETS = [
    dict(
        run_name='augmix_afa_c10_200ep_eval',
        ckpt_path=f'{WANDB_DIR}/offline-run-20260730_130508-oayl9tc0/files/model_best.ckpt',
        config_kwargs=dict(
            ds='c10', m='rn18_dubn',
            attack='afa', use_augmix=True, use_prime=False,
            use_fourier=False, use_wca=False,
            use_apr=False, in_mix=False, use_jsd=False, use_mix=False,
            premix='none', min_str=0., mean_str=5.,
        ),
    ),
    dict(
        run_name='apr_s_c10_200ep_eval',
        ckpt_path=f'{WANDB_DIR}/offline-run-20260730_161322-alx1wa9d/files/model_best.ckpt',
        config_kwargs=dict(
            ds='c10', m='rn18_dubn',
            attack='apr', use_augmix=False, use_prime=False,
            use_fourier=False, use_wca=False,
            use_apr=False, in_mix=False, use_jsd=False, use_mix=False,
            premix='none', min_str=0., mean_str=5.,
        ),
    ),
]

for target in EVAL_TARGETS:
    config = ConfigBuilder.build(**target['config_kwargs'])
    config.data_dir = DATA
    config.num_workers = 4
    config.project = 'afa-wca-c10'
    config.run_name = target['run_name']

    print(f'\n{"=" * 60}')
    print(f'  Eval: {target["run_name"]}')
    print(f'  ckpt: {target["ckpt_path"]}')
    print(f'{"=" * 60}')

    test_accs = eval_only_main(config, ckpt_path=target['ckpt_path'])
    clean_acc = test_accs['clean'][0]
    corr_avg = sum(
        sum(v for v in sev.values()) / len(sev)
        for k, sev in test_accs.items() if k != 'clean'
    ) / (len(test_accs) - 1)
    print(f'  clean={clean_acc:.4f}  corr_avg={corr_avg:.4f}')
