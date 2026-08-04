"""
WaveletBasisSwapOnline: AFA の GeneralFourierOnline と同一インターフェースの
Tensor-native WCA基底スワップ拡張。

forward(x): Tensor[0,1] (B,C,H,W) or (C,H,W) → 同 shape Tensor[0,1]

内部は pywt (NumPy) で処理するため CPU 上で動作する。
GPU テンソルが渡された場合は CPU で処理して元のデバイスに戻す。
"""

import random
import numpy as np
import torch
import torch.nn as nn
import pywt

BASIS_POOL = ["haar", "db4", "db8", "sym4", "sym8", "coif2"]


class WaveletBasisSwapOnline(nn.Module):
    """AFA の GeneralFourierOnline と同一インターフェース。

    Args:
        source_wavelet: 分解に使う基底 (default: 'haar')
        target_wavelet: HF係数の再構成に使う基底 (default: 'db8')
        level: DWT レベル (default: 1)
        swap_prob: 各HF係数をtarget基底でスワップする確率 (default: 0.2)
        mode: pywt の境界拡張モード (default: 'periodization')
    """

    def __init__(
        self,
        source_wavelet: str = 'haar',
        target_wavelet: str = 'db8',
        level: int = 1,
        swap_prob: float = 0.2,
        mode: str = 'periodization',
        swap_ll: bool = False,
    ):
        super().__init__()
        self.source_wavelet = source_wavelet
        self.target_wavelet = target_wavelet
        self.level = level
        self.swap_prob = swap_prob
        self.mode = mode
        self.swap_ll = swap_ll

    def _process_channel(self, ch: np.ndarray, src_w: str, tgt_w: str) -> np.ndarray:
        """1チャンネル (H,W) numpy float64 → 基底スワップ後 numpy float64 (同 shape)。"""
        h, w = ch.shape
        coeffs = pywt.wavedec2(ch, wavelet=src_w, level=self.level, mode=self.mode)
        cA = coeffs[0]
        detail_levels = coeffs[1:]
        zeros_cA = np.zeros_like(cA)

        # LL: swap_ll=True のとき HF と同じ確率で基底スワップ
        ll_w = (tgt_w if random.random() < self.swap_prob else src_w) if self.swap_ll else src_w
        zeros_details = [
            (np.zeros_like(d[0]), np.zeros_like(d[1]), np.zeros_like(d[2]))
            for d in detail_levels
        ]
        result = pywt.waverec2([cA] + zeros_details, wavelet=ll_w, mode=self.mode)

        # 各レベルの各HF係数 (LH, HL, HH) を独立に swap
        for lvl_i, (LH, HL, HH) in enumerate(detail_levels):
            z = [
                (np.zeros_like(d[0]), np.zeros_like(d[1]), np.zeros_like(d[2]))
                for d in detail_levels
            ]
            for coef_i, coef in enumerate([LH, HL, HH]):
                w_use = tgt_w if random.random() < self.swap_prob else src_w
                details = [list(z[i]) for i in range(len(detail_levels))]
                details[lvl_i][coef_i] = coef
                details = [tuple(d) for d in details]
                recon = pywt.waverec2([zeros_cA] + details, wavelet=w_use, mode=self.mode)
                result = result + recon

        return result[:h, :w]

    def _augment_single(self, x_np: np.ndarray) -> np.ndarray:
        """x_np: (C,H,W) float64 [0,1] → (C,H,W) float64 [0,1]"""
        c, h, w = x_np.shape
        out = np.empty_like(x_np)
        for ci in range(c):
            # [0,1] のまま処理（pywt は値域非依存）
            out[ci] = self._process_channel(x_np[ci], self.source_wavelet, self.target_wavelet)
        return np.clip(out, 0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float32 Tensor, shape (C,H,W) or (B,C,H,W), values in [0,1]
        Returns:
            Tensor same shape/dtype/device, values in [0,1]
        """
        init_shape = x.shape
        squeeze = x.ndim == 3
        if squeeze:
            x = x.unsqueeze(0)

        device = x.device
        x_cpu = x.detach().cpu().numpy().astype(np.float64)  # (B,C,H,W)

        out = np.stack([self._augment_single(x_cpu[i]) for i in range(x_cpu.shape[0])])

        result = torch.from_numpy(out).float().to(device)
        if squeeze:
            result = result.squeeze(0)
        return result.reshape(init_shape)

    def __str__(self):
        return (
            f'WaveletBasisSwapOnline('
            f'src={self.source_wavelet}, tgt={self.target_wavelet}, '
            f'level={self.level}, swap_prob={self.swap_prob})'
        )
