"""
APR 系列3実験のキュー:
  1. APR+AFA  : M=APR(p=0.6), A=AFA,  AdvModule, rn18_dubn, 200ep
  2. APR+WCA  : M=APR(p=0.6), A=WCA,  AdvModule, rn18_dubn, 200ep
  3. APR plain: cat([x,APR(x)]),       APRPModule, rn18,     200ep
"""
import os
os.environ['WANDB_MODE'] = 'offline'

from config_utils import ConfigBuilder
from main import main

DATA = '/home/kairisasaki/data/cifar10'
COMMON = dict(
    ds='c10', in_mix=False, use_jsd=False, use_mix=False,
    use_prime=False, use_augmix=False, premix='none',
    min_str=0., mean_str=5.,
)

# ── 1. APR+AFA ──────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('  APR+AFA  (M=APR(p=0.6), A=AFA, AdvModule, rn18_dubn)')
print('=' * 60)
cfg = ConfigBuilder.build(
    m='rn18_dubn', attack='afa',
    use_fourier=False,  # train_aug に AFA は入れない（A-route のみ）
    use_wca=False,
    use_apr=True,       # train_aug に APR(p=0.6)
    **COMMON,
)
cfg.data_dir    = DATA
cfg.num_workers = 4
cfg.project     = 'afa-wca-c10'
cfg.run_name    = 'apr_afa_c10_200ep'
cfg.epochs      = 200
main(cfg)

# ── 2. APR+WCA ──────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('  APR+WCA  (M=APR(p=0.6), A=WCA, AdvModule, rn18_dubn)')
print('=' * 60)
cfg = ConfigBuilder.build(
    m='rn18_dubn', attack='wca',
    use_fourier=False,
    use_wca=False,      # train_aug に WCA は入れない（A-route のみ）
    use_apr=True,       # train_aug に APR(p=0.6)
    **COMMON,
)
cfg.data_dir    = DATA
cfg.num_workers = 4
cfg.project     = 'afa-wca-c10'
cfg.run_name    = 'apr_wca_c10_200ep'
cfg.epochs      = 200
main(cfg)

# ── 3. APR plain (APRPModule) ────────────────────────────────────────────────
print('\n' + '=' * 60)
print('  APR plain  (APRPModule: cat([x, APR(x)]), rn18, 250ep)')
print('=' * 60)
cfg = ConfigBuilder.build(
    m='rn18', attack='apr',
    use_fourier=False,
    use_wca=False,
    use_apr=False,      # train_aug には入れない（attack 側のみ）
    **COMMON,
)
cfg.data_dir    = DATA
cfg.num_workers = 4
cfg.project     = 'afa-wca-c10'
cfg.run_name    = 'apr_plain_c10_200ep'
cfg.epochs      = 200
main(cfg)

print('\n=== APR combo queue 完了 ===')
