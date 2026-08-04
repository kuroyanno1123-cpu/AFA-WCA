"""
APR+WCA-LL スイープ + ベースライン キュー

実行順:
  1. APR+WCA-LL  p=0.2   (rn18_dubn, AdvModule, use_apr=True, 200ep)
  2. 比較: APR+WCA p=0.2 (mCE=0.1678) vs APR+WCA-LL p=0.2 → winner 決定
  3. winner の p=0.3, 0.4, 0.5 スイープ
  4. Baseline (rn18, no attack, no aug, 200ep)
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

APR_WCA_MCE15 = 0.1678  # apr_wca_c10_200ep の既存結果


def _read_mce(run_name):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'results', run_name, 'cc_results.txt',
    )
    with open(path) as f:
        for line in f:
            if line.startswith('mCE='):
                return float(line.strip().split('=')[1])
    raise RuntimeError(f'mCE not found in {path}')


def _run_wca(swap_prob, swap_ll, run_name, epochs=200):
    cfg = ConfigBuilder.build(
        m='rn18_dubn', attack='wca',
        use_fourier=False,
        use_wca=False,
        use_apr=True,
        wca_swap_prob=swap_prob,
        wca_swap_ll=swap_ll,
        **COMMON,
    )
    cfg.data_dir    = DATA
    cfg.num_workers = 4
    cfg.project     = 'afa-wca-c10'
    cfg.run_name    = run_name
    cfg.epochs      = epochs
    main(cfg)


# ── 1. APR+WCA-LL p=0.2 ─────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('  APR+WCA-LL  p=0.2  (wca_swap_ll=True)')
print('=' * 60)
_run_wca(0.2, True, 'apr_wca_ll_c10_p02_200ep')

# ── 2. 比較・winner 決定 ─────────────────────────────────────────────────────
mce_ll_02 = _read_mce('apr_wca_ll_c10_p02_200ep')
print('\n' + '=' * 60)
print(f'  APR+WCA    p=0.2 mCE = {APR_WCA_MCE15:.4f}')
print(f'  APR+WCA-LL p=0.2 mCE = {mce_ll_02:.4f}')

use_ll = mce_ll_02 < APR_WCA_MCE15
winner = 'APR+WCA-LL' if use_ll else 'APR+WCA'
winner_swap_ll = use_ll
print(f'  → winner: {winner}  (swap_ll={winner_swap_ll})')
print('=' * 60)

# ── 3. winner を p=0.3, 0.4, 0.5 でスイープ ─────────────────────────────────
for p_str, p_val in [('03', 0.3), ('04', 0.4), ('05', 0.5)]:
    tag = 'll' if winner_swap_ll else 'std'
    run_name = f'apr_wca_{tag}_c10_p{p_str}_200ep'
    print('\n' + '=' * 60)
    print(f'  {winner}  p={p_val}  (swap_ll={winner_swap_ll})')
    print('=' * 60)
    _run_wca(p_val, winner_swap_ll, run_name)

# ── 4. Baseline ──────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('  Baseline  (rn18, no attack, no aug, 200ep)')
print('=' * 60)
cfg = ConfigBuilder.build(
    m='rn18', attack='none',
    use_fourier=False,
    use_wca=False,
    use_apr=False,
    use_prime=False,
    use_augmix=False,
    ds='c10', in_mix=False, use_jsd=False, use_mix=False,
    premix='none',
)
cfg.data_dir    = DATA
cfg.num_workers = 4
cfg.project     = 'afa-wca-c10'
cfg.run_name    = 'baseline_c10_200ep'
cfg.epochs      = 200
main(cfg)

print('\n=== wca_ll_sweep_baseline queue 完了 ===')
