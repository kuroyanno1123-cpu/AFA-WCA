"""
Step 3 テスト:
  1. WaveletBasisSwapOnline: shape/dtype/値域[0,1] 保持
  2. swap_prob=0 でほぼ恒等変換
  3. WCA選択時の1バッチ forward→ACE損失
  4. AFA既存パス (GeneralFourierOnline) に回帰なし
  5. CE_main / CE_aux 分離ログ確認
"""
import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np

DATA_DIR = '/home/kairisasaki/data/cifar10'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Test 1: shape / dtype / 値域 [0,1] ─────────────────────────────────────
def test_shape_dtype_range():
    from project.augs.wca import WaveletBasisSwapOnline
    aug = WaveletBasisSwapOnline(swap_prob=0.5)

    # (B,C,H,W)
    x = torch.rand(4, 3, 32, 32, device=DEVICE)
    y = aug(x)
    assert y.shape == x.shape, f'shape mismatch: {y.shape} != {x.shape}'
    assert y.dtype == torch.float32, f'dtype: {y.dtype}'
    assert y.min() >= 0.0 and y.max() <= 1.0, f'range: [{y.min():.4f}, {y.max():.4f}]'

    # (C,H,W)
    x1 = torch.rand(3, 32, 32, device=DEVICE)
    y1 = aug(x1)
    assert y1.shape == x1.shape, f'shape (C,H,W) mismatch: {y1.shape}'

    print(f'  PASS  test_shape_dtype_range  y.range=[{y.min():.4f},{y.max():.4f}]')


# ── Test 2: swap_prob=0 → ほぼ恒等変換 ─────────────────────────────────────
def test_identity_at_zero_swap():
    from project.augs.wca import WaveletBasisSwapOnline
    aug = WaveletBasisSwapOnline(swap_prob=0.0)

    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32)
    y = aug(x)
    # DWT→IDWT ラウンドトリップ: periodization モードでほぼ完全一致
    max_diff = (y - x).abs().max().item()
    assert max_diff < 1e-4, f'swap_prob=0 should be near-identity, max_diff={max_diff:.2e}'
    print(f'  PASS  test_identity_at_zero_swap  max_diff={max_diff:.2e}')


# ── Test 3: WCA選択時 1バッチ forward → ACE損失 backward ──────────────────
def test_wca_forward_backward():
    import torchvision.transforms as T
    from project.models.image_classification import get_model
    from project.dsets.vision.dataset import CIFAR10Dataset
    from project.augs.wca import WaveletBasisSwapOnline
    from project.dsets import get_dataset
    from torch.utils.data import DataLoader

    dataset_cls = get_dataset('c10')
    normalise = T.Normalize(mean=dataset_cls.mean, std=dataset_cls.std)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    dset = CIFAR10Dataset(root=DATA_DIR, train=True, transform=train_tf, download=False)
    dl   = DataLoader(dset, batch_size=8, shuffle=True, num_workers=0)

    model_cls = get_model('c10', 'rn18')
    model = model_cls(num_classes=10).to(DEVICE)
    aug = WaveletBasisSwapOnline(source_wavelet='haar', target_wavelet='db8',
                                  level=1, swap_prob=0.2)

    x, y = next(iter(dl))
    x, y = x.to(DEVICE), y.to(DEVICE)

    model.train()
    x_aux = aug(x)
    logits_main = model(normalise(x))
    logits_aux  = model(normalise(x_aux))

    ce_main = F.cross_entropy(logits_main, y)
    ce_aux  = F.cross_entropy(logits_aux,  y)
    loss    = (ce_main + ce_aux) / 2.0

    loss.backward()
    grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5

    assert not torch.isnan(loss), 'loss is NaN'
    assert grad_norm > 0, 'grad_norm == 0'

    print(f'  PASS  test_wca_forward_backward'
          f'  CE_main={ce_main.item():.4f}  CE_aux={ce_aux.item():.4f}'
          f'  ACE={loss.item():.4f}  grad_norm={grad_norm:.4f}')


# ── Test 4: AFA既存パス (GeneralFourierOnline) 回帰確認 ────────────────────
def test_afa_fba_no_regression():
    from project.augs.fba.fourier_basis import GeneralFourierOnline
    aug = GeneralFourierOnline(
        img_size=32, groups=list(range(1,16)), phases=(0,1),
        f_cut=1, phase_cut=1, min_str=0, mean_str=5, granularity=64
    )
    x = torch.rand(4, 3, 32, 32, device=DEVICE)
    y = aug(x)
    assert y.shape == x.shape
    assert y.min() >= 0.0 and y.max() <= 1.0
    print(f'  PASS  test_afa_fba_no_regression  y.range=[{y.min():.4f},{y.max():.4f}]')


# ── Test 5: CE_main / CE_aux 分離ログ確認 ──────────────────────────────────
def test_ce_logging():
    from project.augs.wca import WaveletBasisSwapOnline
    from project.models.image_classification import get_model
    from project.dsets import get_dataset
    import torchvision.transforms as T
    from project.dsets.vision.dataset import CIFAR10Dataset
    from torch.utils.data import DataLoader

    dataset_cls = get_dataset('c10')
    normalise = T.Normalize(mean=dataset_cls.mean, std=dataset_cls.std)
    train_tf = T.Compose([T.ToTensor()])
    dset  = CIFAR10Dataset(root=DATA_DIR, train=True, transform=train_tf, download=False)
    dl    = DataLoader(dset, batch_size=8, shuffle=False, num_workers=0)
    model = get_model('c10', 'rn18')(num_classes=10).to(DEVICE)
    aug   = WaveletBasisSwapOnline(swap_prob=0.5)

    x, y = next(iter(dl))
    x, y = x.to(DEVICE), y.to(DEVICE)
    x_aux = aug(x)

    model.eval()
    with torch.no_grad():
        ce_main = F.cross_entropy(model(normalise(x)), y).item()
        ce_aux  = F.cross_entropy(model(normalise(x_aux)), y).item()
        ace     = (ce_main + ce_aux) / 2.0

    # CE_main と CE_aux が独立に計算できることを確認
    assert isinstance(ce_main, float) and isinstance(ce_aux, float)
    print(f'  PASS  test_ce_logging'
          f'  CE_main={ce_main:.4f}  CE_aux={ce_aux:.4f}  ACE={ace:.4f}')


# ── runner ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tests = [
        ('shape/dtype/range',          test_shape_dtype_range),
        ('identity at swap_prob=0',    test_identity_at_zero_swap),
        ('WCA forward/backward',       test_wca_forward_backward),
        ('AFA FBA no regression',      test_afa_fba_no_regression),
        ('CE_main/CE_aux logging',     test_ce_logging),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f'  FAIL  {name}: {e}')
            traceback.print_exc()
            failed += 1

    print(f'\n{passed}/{passed+failed} tests passed')
    if failed:
        sys.exit(1)
