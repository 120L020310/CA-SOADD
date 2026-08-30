import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# sinc/filter tools
# =========================
def _hann_window(n: int, device=None, dtype=None) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2 * math.pi * k / (n - 1))

def _sinc_lowpass(kernel_size: int, cutoff: float, device=None, dtype=None) -> torch.Tensor:
    cutoff = float(cutoff)
    if cutoff <= 0.0:
        return torch.zeros(kernel_size, device=device, dtype=dtype)
    cutoff = min(0.5, cutoff)

    k = torch.arange(kernel_size, device=device, dtype=dtype)
    center = (kernel_size - 1) / 2.0
    n = k - center
    h = 2 * cutoff * torch.sinc(2 * cutoff * n)
    h = h * _hann_window(kernel_size, device=device, dtype=dtype)
    h = h / (h.sum() + 1e-8)
    return h

def _bandpass_bank(num_bands: int, kernel_size: int, device=None, dtype=None) -> torch.Tensor:
    edges = torch.linspace(0.0, 0.5, num_bands + 1, device=device, dtype=dtype)
    lps = []
    for fc in edges:
        lps.append(_sinc_lowpass(kernel_size, float(fc.item()), device=device, dtype=dtype))
    lps = torch.stack(lps, dim=0)          # (K+1, L)
    bank = lps[1:] - lps[:-1]              # (K, L)
    return bank

# =========================
# Pseudo Neural Codec Resynthesis
# =========================
class PseudoNeuralCodecResynthesis(nn.Module):
    def __init__(
        self,
        sr=16000, num_bands=8, analysis_kernel=63, stride=4,
        bits=6, mu=255.0, boundary_hop_ms=20.0, boundary_strength=0.03,
        recon_kernel=9, recon_cutoff=0.48, quant_mode="ste"
    ):
        super().__init__()
        assert analysis_kernel % 2 == 1
        assert recon_kernel % 2 == 1
        assert stride >= 2
        assert quant_mode in ("ste", "none", "detach")

        self.sr = int(sr)
        self.num_bands = int(num_bands)
        self.analysis_kernel = int(analysis_kernel)
        self.stride = int(stride)
        self.bits = int(bits)
        self.mu = float(mu)
        self.boundary_hop_ms = float(boundary_hop_ms)
        self.boundary_strength = float(boundary_strength)
        self.quant_mode = quant_mode
        self.strength = 1.0

        bank = _bandpass_bank(self.num_bands, self.analysis_kernel, dtype=torch.float32)
        self.register_buffer("analysis_bank", bank.view(self.num_bands, 1, self.analysis_kernel))

        recon = _sinc_lowpass(recon_kernel, recon_cutoff, dtype=torch.float32)
        self.register_buffer("recon_kernel", recon.view(1, 1, recon_kernel))

        self.register_buffer("up_weight", torch.ones(self.num_bands, 1, 1, dtype=torch.float32))

    def _quantize(self, y: torch.Tensor) -> torch.Tensor:
        if self.quant_mode == "none":
            return y
        levels = (1 << self.bits) - 1
        y01 = (y + 1.0) * 0.5
        yq = torch.round(y01 * levels) / levels
        yq = yq * 2.0 - 1.0

        if self.quant_mode == "detach":
            return yq.detach()
        return y + (yq - y).detach()  # STE

    @staticmethod
    def _crop_or_pad(y: torch.Tensor, T: int) -> torch.Tensor:
        L = y.shape[-1]
        if L > T:
            return y[..., :T]
        if L < T:
            return F.pad(y, (0, T - L))
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,T)
        B, C, T = x.shape
        w = self.analysis_bank.to(device=x.device, dtype=x.dtype)
        pad = (self.analysis_kernel - 1) // 2

        sub = F.conv1d(x, w, stride=self.stride, padding=pad)  # (B,K,Ls)

        sub_c = torch.sign(sub) * torch.log1p(self.mu * torch.abs(sub)) / math.log1p(self.mu)
        sub_q = self._quantize(sub_c)
        sub_e = torch.sign(sub_q) * (torch.expm1(torch.abs(sub_q) * math.log1p(self.mu)) / self.mu)

        Ls = sub_e.shape[-1]
        hop_sub = max(1, int(round((self.boundary_hop_ms / 1000.0) * self.sr / self.stride)))
        idx = torch.arange(Ls, device=x.device, dtype=x.dtype)
        phase = 2 * math.pi * (idx % hop_sub) / hop_sub
        env = 1.0 + self.boundary_strength * (torch.cos(phase) ** 3)
        sub_e = sub_e * env.view(1, 1, Ls)

        up_w = self.up_weight.to(device=x.device, dtype=x.dtype)
        y = F.conv_transpose1d(sub_e, up_w, stride=self.stride, groups=self.num_bands)

        rk = self.recon_kernel.to(device=x.device, dtype=x.dtype).repeat(self.num_bands, 1, 1)
        pad_r = (rk.shape[-1] - 1) // 2
        y = F.conv1d(y, rk, padding=pad_r, groups=self.num_bands)

        y = y.sum(dim=1, keepdim=True)
        y = self._crop_or_pad(y, T)
        return x + self.strength * (y - x)

# =========================
# Weak vocoder artifacts
# =========================
class WeakVocoderHFArtifacts(nn.Module):
    def __init__(
        self,
        sr=16000, split_lp_cutoff=0.18, split_kernel=63,
        comb_stride=4, comb_strength=0.08,
        phase_cycles=1.5, phase_max_delay=0.25,
        hf_smooth_kernel=5, smooth_strength=0.15, strength=1.0
    ):
        super().__init__()
        self.sr = int(sr)
        self.comb_stride = int(comb_stride)
        self.comb_strength = float(comb_strength)
        self.phase_cycles = float(phase_cycles)
        self.phase_max_delay = float(phase_max_delay)
        self.smooth_strength = float(smooth_strength)
        self.strength = float(strength)

        lp = _sinc_lowpass(split_kernel, split_lp_cutoff, dtype=torch.float32)
        self.register_buffer("lp_kernel", lp.view(1, 1, split_kernel))

        if hf_smooth_kernel == 5:
            sm = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
            sm = (sm / sm.sum()).view(1, 1, -1)
        else:
            w = _hann_window(hf_smooth_kernel, dtype=torch.float32)
            sm = (w / (w.sum() + 1e-8)).view(1, 1, -1)
        self.register_buffer("hf_smooth", sm)

    @staticmethod
    def _shift_right(x: torch.Tensor, n: int) -> torch.Tensor:
        if n <= 0:
            return x
        B, C, T = x.shape
        return torch.cat([torch.zeros(B, C, n, device=x.device, dtype=x.dtype), x[..., :T - n]], dim=-1)

    @staticmethod
    def _shift_left(x: torch.Tensor, n: int) -> torch.Tensor:
        if n <= 0:
            return x
        B, C, T = x.shape
        return torch.cat([x[..., n:], torch.zeros(B, C, n, device=x.device, dtype=x.dtype)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,T)
        B, C, T = x.shape
        lp = self.lp_kernel.to(device=x.device, dtype=x.dtype)
        pad = (lp.shape[-1] - 1) // 2

        low = F.conv1d(x, lp, padding=pad)
        high = x - low

        s = self.comb_stride
        if s >= 1 and self.comb_strength != 0.0:
            delayed = self._shift_right(high, s)
            high = high + self.comb_strength * (high - delayed)

        t = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype).view(1, 1, T)
        delta = self.phase_max_delay * torch.sin(2 * math.pi * self.phase_cycles * t)
        delta = delta.clamp(-0.49, 0.49)

        hf_fwd = self._shift_left(high, 1)
        hf_bwd = self._shift_right(high, 1)
        dp = delta.clamp(min=0.0)
        dn = (-delta).clamp(min=0.0)
        high = high + dp * (hf_fwd - high) + dn * (hf_bwd - high)

        sm = self.hf_smooth.to(device=x.device, dtype=x.dtype)
        pad2 = (sm.shape[-1] - 1) // 2
        high_sm = F.conv1d(high, sm, padding=pad2)
        high = (1.0 - self.smooth_strength) * high + self.smooth_strength * high_sm

        y = low + high
        return x + self.strength * (y - x)

# =========================
# ID-invariant augment
# =========================
class IDInvariantAugment(nn.Module):
    def __init__(
        self,
        sr=16000, lp_kernel=63, cutoffs=(0.20, 0.25, 0.30, 0.35, 0.40),
        noise_snr_db=(28.0, 40.0), clip_gain=(1.2, 2.5), strength=1.0
    ):
        super().__init__()
        assert lp_kernel % 2 == 1
        self.sr = int(sr)
        self.noise_snr_db = noise_snr_db
        self.clip_gain = clip_gain
        self.strength = float(strength)

        kernels = []
        for fc in cutoffs:
            kernels.append(_sinc_lowpass(lp_kernel, float(fc), dtype=torch.float32))
        bank = torch.stack(kernels, dim=0)  # (M, L)
        self.register_buffer("lp_bank", bank.view(len(cutoffs), 1, lp_kernel))

    def _bandlimit(self, x: torch.Tensor) -> torch.Tensor:
        M = self.lp_bank.shape[0]
        idx = int(torch.randint(0, M, (1,), device=x.device).item())
        w = self.lp_bank[idx:idx + 1].to(device=x.device, dtype=x.dtype)
        pad = (w.shape[-1] - 1) // 2
        return F.conv1d(x, w, padding=pad)

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self.noise_snr_db
        r = torch.rand((), device=x.device, dtype=x.dtype)
        snr = lo + (hi - lo) * r

        p = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8)
        noise = torch.randn_like(x)
        pn = noise.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.sqrt(p / (pn * (10 ** (snr / 10.0))))
        return x + noise * scale

    def _soft_clip(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self.clip_gain
        g = lo + (hi - lo) * torch.rand((), device=x.device, dtype=x.dtype)
        return torch.tanh(g * x) / torch.tanh(g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = int(torch.randint(0, 3, (1,), device=x.device).item())
        if op == 0:
            y = self._bandlimit(x)
        elif op == 1:
            y = self._add_noise(x)
        else:
            y = self._soft_clip(x)
        return x + self.strength * (y - x)

# =========================
# AudioPermutation
# =========================
class AudioPermutation(nn.Module):
    def __init__(self, n_segments=8):
        super().__init__()
        self.n_segments = int(n_segments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,T) or (B,T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, C, T = x.shape
        n = self.n_segments
        seg_len = T // n
        T2 = seg_len * n
        x2 = x[..., :T2]  # 裁掉不整除部分（也可改成pad）
        segments = x2.view(B, C, n, seg_len)
        idx = torch.randperm(n, device=x.device)
        y = segments[:, :, idx, :].reshape(B, C, T2)
        if T2 < T:
            y = F.pad(y, (0, T - T2))
        return y

# =========================
# AudioSpectralMasking
# =========================

class AudioSpectralMasking(nn.Module):
    def __init__(self, mask_ratio=0.1):
        super().__init__()
        self.mask_ratio = float(mask_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, C, T = x.shape
        spec = torch.fft.rfft(x, dim=-1)  # (B,C,F)
        Fbins = spec.shape[-1]

        mask_len = max(1, int(Fbins * self.mask_ratio))
        start = int(torch.randint(0, max(1, Fbins - mask_len + 1), (1,), device=x.device).item())

        mask = torch.ones_like(spec)
        mask[..., start:start + mask_len] = 0
        spec2 = spec * mask
        y = torch.fft.irfft(spec2, n=T, dim=-1)
        return y

# =========================
# AudioPhaseScrambling
# =========================

class AudioPhaseScrambling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, C, T = x.shape
        spec = torch.fft.rfft(x, dim=-1)
        amp = torch.abs(spec)
        # 随机相位：exp(j*theta)
        theta = 2 * math.pi * torch.rand_like(amp)
        rnd_phase = torch.cos(theta) + 1j * torch.sin(theta)
        spec2 = amp * rnd_phase
        y = torch.fft.irfft(spec2, n=T, dim=-1)
        return y

# =========================
# AudioResampleShift
# =========================

class AudioResampleShift(nn.Module):
    def __init__(self, orig_sr=48000, target_sr=16000):
        super().__init__()
        self.orig_sr = int(orig_sr)
        self.target_sr = int(target_sr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, C, T = x.shape
        down_len = max(4, int(round(T * self.target_sr / self.orig_sr)))
        y = F.interpolate(x, size=down_len, mode="linear", align_corners=False)
        y = F.interpolate(y, size=T, mode="linear", align_corners=False)
        return y


class Distribution_shifter(nn.Module):
    """
      0 PNCR
      1 WeakVocoderHFArtifacts
      2 IDInvariantAugment
      3 Permutation
      4 SpectralMask
      5 PhaseScramble
      6 ResampleShift
    """
    def __init__(
        self,
        sr: int = 48000,
        target_len: Optional[int] = None,
        selection_mode: str = "batch",   # "batch" or "per_sample"
        mix: float = 1.0,
        detach_aug: bool = False,
        clamp: bool = False,

        # second-set params
        perm_segments: int = 8,
        spec_mask_ratio: float = 0.1,
        resample_target_sr: int = 16000,
    ):
        super().__init__()
        assert selection_mode in ("batch", "per_sample")
        assert 0.0 <= mix <= 1.0

        self.sr = int(sr)
        self.target_len = target_len
        self.selection_mode = selection_mode
        self.mix = float(mix)
        self.detach_aug = bool(detach_aug)
        self.clamp = bool(clamp)

        # --- first set ---
        self.pncr = PseudoNeuralCodecResynthesis(sr=sr)
        self.vocoder = WeakVocoderHFArtifacts(sr=sr)
        self.idinv = IDInvariantAugment(sr=sr)

        # --- second set ---
        self.perm = AudioPermutation(n_segments=perm_segments)
        self.specmask = AudioSpectralMasking(mask_ratio=spec_mask_ratio)
        self.phases = AudioPhaseScrambling()
        self.resamp = AudioResampleShift(orig_sr=sr, target_sr=resample_target_sr)

        self.ops: List[nn.Module] = nn.ModuleList([
            self.pncr,
            self.vocoder,
            self.idinv,
            self.perm,
            self.specmask,
            self.phases,
            self.resamp
        ])

    @staticmethod
    def _ensure_2d(wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() != 2:
            raise ValueError(f"Expected (B,T), got {tuple(wav.shape)}")
        return wav

    def _ensure_len(self, wav: torch.Tensor) -> torch.Tensor:
        if self.target_len is None:
            return wav
        B, T = wav.shape
        L = int(self.target_len)
        if T == L:
            return wav
        if T < L:
            return F.pad(wav, (0, L - T))
        # T > L
        if self.training:
            start = int(torch.randint(0, T - L + 1, (1,), device=wav.device).item())
        else:
            start = (T - L) // 2
        return wav[:, start:start + L]

    def _dispatch(self, x1: torch.Tensor, idx: int) -> torch.Tensor:
        return self.ops[idx](x1)

    def forward(self, wav: torch.Tensor, return_idx: bool = False):
        wav = self._ensure_2d(wav)
        wav = self._ensure_len(wav)

        B, T = wav.shape
        x1 = wav.unsqueeze(1)  # (B,1,T)
        K = len(self.ops)

        if self.selection_mode == "batch":
            idx = int(torch.randint(0, K, (1,), device=wav.device).item())
            y1 = self._dispatch(x1, idx)
            y = y1.squeeze(1)
            idx_out = torch.full((B,), idx, device=wav.device, dtype=torch.long)
        else:
            idx = torch.randint(0, K, (B, 1, 1), device=wav.device)
            ys = []
            for k in range(K):
                ys.append(self._dispatch(x1, k).squeeze(1))
            ys = torch.stack(ys, dim=1)  # (B,K,T)
            y = ys.gather(1, idx.expand(-1, 1, T)).squeeze(1)
            idx_out = idx.view(B)

        # mix
        if self.mix < 1.0:
            y = (1.0 - self.mix) * wav + self.mix * y

        # stop-grad through augmentation
        if self.detach_aug:
            y = wav + (y - wav).detach()

        # ensure len again
        y = self._ensure_len(y)

        if self.clamp:
            y = y.clamp(-1.0, 1.0)

        if return_idx:
            return y, idx_out
        return y

# =========================
# Example
# =========================
# shifter = UnifiedRandomAudioShifter(
#     sr=48000, target_len=48000,
#     selection_mode="batch", mix=1.0,
#     perm_segments=8, spec_mask_ratio=0.1, resample_target_sr=16000
# ).cuda()
# wav = torch.randn(8, 48000, device="cuda", requires_grad=True)
# y, idx = shifter(wav, return_idx=True)
# y.mean().backward()
# print(y.shape, idx[:4], wav.grad.abs().mean().item())
