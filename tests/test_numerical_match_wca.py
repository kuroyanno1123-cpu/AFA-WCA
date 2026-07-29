"""
WaveletBasisSwapOnline (AFA-WCA) vs WaveletBasisSwap (WCA original) 数値一致テスト

比較戦略:
  - DWT/IDWT は線形変換なので入力スケールに比例して出力もスケールする
  - 元実装: PIL Image [0,255] → _process_channel → float64 [0,255]
  - 本実装: Tensor [0,1]     → _process_channel → float64 [0,1]
  - 比較:  orig_out / 255.0  vs  mine_out  で allclose を検証
  - uint8 変換前の float64 中間値で比較（量子化誤差を除外）

テスト一覧:
  T1: _process_channel 単体（同シード・同入力）
  T2: チャンネルループ全体（同シード・同入力）
  T3: PIL → Tensor 変換込みの end-to-end (uint8 量子化誤差内で一致)
  T4: swap_prob=0 と swap_prob=1.0 の極端ケース
  T5: HF係数加算合成の確認（係数ごとに独立に再構成して加算するロジックの同一性）
"""

import sys, os, traceback, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/kairisasaki/WCA')

import numpy as np
import torch
from PIL import Image

# ── 両実装をインポート ──────────────────────────────────────────────────────
from project.augs.wca.wavelet_basis_swap import WaveletBasisSwapOnline
from core.wca import WaveletBasisSwap

SRC_W  = 'haar'
TGT_W  = 'db8'
LEVEL  = 1
MODE   = 'periodization'
H, W   = 32, 32
SEED   = 42


def make_test_image(seed=0):
    """再現性のある 32x32 テスト画像を返す (PIL, numpy [0,255], numpy [0,1], tensor [0,1])"""
    rng = np.random.default_rng(seed)
    arr_uint8 = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    pil_img   = Image.fromarray(arr_uint8)
    arr_f64   = arr_uint8.astype(np.float64)                 # HWC float64 [0,255]
    arr_f64_01 = arr_f64.transpose(2, 0, 1) / 255.0         # CHW float64 [0,1] ← float32 経由なし
    tensor    = torch.from_numpy(arr_f64_01.astype(np.float32))  # CHW float32 (実使用を模擬)
    return pil_img, arr_f64, arr_f64_01, tensor


# ── T1: _process_channel 単体の数値一致 ────────────────────────────────────
def test_process_channel_match():
    orig = WaveletBasisSwap(source_wavelet=SRC_W, target_wavelet=TGT_W,
                            level=LEVEL, swap_prob=0.5, mode=MODE)
    mine = WaveletBasisSwapOnline(source_wavelet=SRC_W, target_wavelet=TGT_W,
                                  level=LEVEL, swap_prob=0.5, mode=MODE)

    _, arr_f64, arr_f64_01, _ = make_test_image(seed=0)
    ch_255 = arr_f64[:, :, 0]        # R チャンネル, float64 [0,255]
    ch_01  = arr_f64_01[0]           # R チャンネル, float64 [0,1]  ← float32経由なし

    random.seed(SEED)
    out_orig = orig._process_channel(ch_255, SRC_W, TGT_W)  # float64, ~[0,255]

    random.seed(SEED)
    out_mine = mine._process_channel(ch_01, SRC_W, TGT_W)   # float64, ~[0,1]

    # DWT は線形なので: _process_channel(x*255) == _process_channel(x) * 255
    diff = np.abs(out_orig / 255.0 - out_mine)
    max_diff = diff.max()
    mean_diff = diff.mean()

    # float64 算術誤差の上限 (~1e-12 程度)
    assert max_diff < 1e-10, (
        f'_process_channel mismatch: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}'
    )
    print(f'  PASS  T1 _process_channel match  max_diff={max_diff:.2e}')


# ── T2: チャンネルループ全体の数値一致 ─────────────────────────────────────
def test_channel_loop_match():
    orig = WaveletBasisSwap(source_wavelet=SRC_W, target_wavelet=TGT_W,
                            level=LEVEL, swap_prob=0.5, mode=MODE)
    mine = WaveletBasisSwapOnline(source_wavelet=SRC_W, target_wavelet=TGT_W,
                                  level=LEVEL, swap_prob=0.5, mode=MODE)

    _, arr_f64, arr_f64_01, _ = make_test_image(seed=7)

    # 元実装: HWC float64 [0,255]
    result_orig = np.zeros_like(arr_f64)
    random.seed(SEED)
    for c in range(3):
        result_orig[:, :, c] = orig._process_channel(arr_f64[:, :, c], SRC_W, TGT_W)

    # 本実装: CHW float64 [0,1] ← float32経由なし
    result_mine = np.empty_like(arr_f64_01)
    random.seed(SEED)
    for ci in range(3):
        result_mine[ci] = mine._process_channel(arr_f64_01[ci], SRC_W, TGT_W)

    # HWC [0,255] → CHW [0,1] に変換して比較
    result_orig_chw = result_orig.transpose(2, 0, 1) / 255.0
    diff = np.abs(result_orig_chw - result_mine)
    max_diff = diff.max()
    assert max_diff < 1e-10, f'channel loop mismatch: max_diff={max_diff:.2e}'
    print(f'  PASS  T2 channel loop match  max_diff={max_diff:.2e}')


# ── T3: end-to-end (uint8量子化誤差込み) ───────────────────────────────────
def test_end_to_end_match():
    orig = WaveletBasisSwap(source_wavelet=SRC_W, target_wavelet=TGT_W,
                            level=LEVEL, swap_prob=0.5, mode=MODE)
    mine = WaveletBasisSwapOnline(source_wavelet=SRC_W, target_wavelet=TGT_W,
                                  level=LEVEL, swap_prob=0.5, mode=MODE)

    pil_img, _, arr_f64_01, tensor = make_test_image(seed=3)

    # 元実装: PIL → PIL (uint8 clip)
    random.seed(SEED)
    out_pil = orig(pil_img)
    out_orig_arr = np.array(out_pil).astype(np.float64) / 255.0  # HWC [0,1]

    # 本実装: Tensor [0,1] → Tensor [0,1]
    random.seed(SEED)
    out_tensor = mine(tensor.unsqueeze(0)).squeeze(0)  # CHW
    out_mine_arr = out_tensor.numpy()                  # CHW [0,1]

    # HWC vs CHW → 転置して比較（uint8量子化誤差: max 0.5/255 ≒ 2e-3）
    out_orig_chw = out_orig_arr.transpose(2, 0, 1)
    diff = np.abs(out_orig_chw - out_mine_arr)
    max_diff = diff.max()

    # uint8量子化誤差の理論上限: 0.5/255 ≈ 0.00196
    assert max_diff < 1.0 / 255.0 + 1e-6, (
        f'end-to-end mismatch exceeds uint8 quantization bound: max_diff={max_diff:.4f}'
    )
    print(f'  PASS  T3 end-to-end match  max_diff={max_diff:.4f} (uint8 quant bound={1/255:.4f})')


# ── T4: 極端な swap_prob (0.0 と 1.0) ──────────────────────────────────────
def test_extreme_swap_prob():
    for prob in [0.0, 1.0]:
        orig = WaveletBasisSwap(source_wavelet=SRC_W, target_wavelet=TGT_W,
                                level=LEVEL, swap_prob=prob, mode=MODE)
        mine = WaveletBasisSwapOnline(source_wavelet=SRC_W, target_wavelet=TGT_W,
                                      level=LEVEL, swap_prob=prob, mode=MODE)

        _, arr_f64, arr_f64_01, _ = make_test_image(seed=5)
        ch_255 = arr_f64[:, :, 1]
        ch_01  = arr_f64_01[1]  # float64 [0,1] ← float32経由なし

        # 乱数消費が同じかを swap_prob=0.0/1.0 で確認
        # (swap_prob が 0 か 1 なら random.random() の値によらず結果は確定)
        random.seed(SEED)
        out_orig = orig._process_channel(ch_255, SRC_W, TGT_W)
        random.seed(SEED)
        out_mine = mine._process_channel(ch_01, SRC_W, TGT_W)

        max_diff = np.abs(out_orig / 255.0 - out_mine).max()
        assert max_diff < 1e-10, f'prob={prob}: max_diff={max_diff:.2e}'
        print(f'  PASS  T4 extreme swap_prob={prob}  max_diff={max_diff:.2e}')


# ── T5: HF係数加算合成の確認 ────────────────────────────────────────────────
def test_additive_hf_synthesis():
    """
    元実装が「係数ごとに独立再構成して加算」することを確認。
    swap_prob=1.0 のとき全HF係数が tgt_w で再構成される。
    手動で再構成した結果と一致するはず。
    """
    import pywt as pywt_
    mine = WaveletBasisSwapOnline(source_wavelet=SRC_W, target_wavelet=TGT_W,
                                  level=LEVEL, swap_prob=1.0, mode=MODE)

    _, _, arr_f64_01, _ = make_test_image(seed=9)
    ch = arr_f64_01[0]  # float64 [0,1]

    # 手動再構成
    coeffs = pywt_.wavedec2(ch, wavelet=SRC_W, level=LEVEL, mode=MODE)
    cA, (LH, HL, HH) = coeffs[0], coeffs[1]
    z = (np.zeros_like(LH), np.zeros_like(HL), np.zeros_like(HH))
    zeros_cA = np.zeros_like(cA)

    # LL term
    manual = pywt_.waverec2([cA, z], wavelet=SRC_W, mode=MODE)
    # HF terms (全て tgt_w で再構成: prob=1.0)
    for coef_i, coef in enumerate([LH, HL, HH]):
        det = [list(z)]
        det[0][coef_i] = coef
        det = [tuple(d) for d in det]
        manual = manual + pywt_.waverec2([zeros_cA] + det, wavelet=TGT_W, mode=MODE)
    manual = manual[:H, :W]

    # mine の _process_channel (prob=1.0 なので random.random() の値は不問)
    out_mine = mine._process_channel(ch, SRC_W, TGT_W)

    max_diff = np.abs(manual - out_mine).max()
    assert max_diff < 1e-10, f'additive synthesis mismatch: max_diff={max_diff:.2e}'
    print(f'  PASS  T5 additive HF synthesis  max_diff={max_diff:.2e}')


# ── runner ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tests = [
        ('T1 _process_channel match',      test_process_channel_match),
        ('T2 channel loop match',          test_channel_loop_match),
        ('T3 end-to-end match (uint8)',    test_end_to_end_match),
        ('T4 extreme swap_prob',           test_extreme_swap_prob),
        ('T5 additive HF synthesis',       test_additive_hf_synthesis),
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
