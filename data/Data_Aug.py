import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
    
class SafeRawAugmentor(nn.Module):
    def __init__(self, noise_intensity=0.05, mask_ratio=0.1):
        super().__init__()
        self.noise_intensity = noise_intensity
        self.mask_ratio = mask_ratio

    def forward(self, audio):
        """
        Input: audio [B, 48000]
        Output: audio_aug [B, 48000]
        """
        B, L = audio.shape
        device = audio.device
        
        audio_aug = audio.clone()

        # ---------------------------------------------------
        # 1. (Amplitude Scaling) 
        # ---------------------------------------------------
        scale = 0.5 + torch.rand(B, 1, device=device)
        audio_aug = audio_aug * scale

        # ---------------------------------------------------
        # 2.(Additive White Gaussian Noise)
        # ---------------------------------------------------
        noise = torch.randn_like(audio_aug)
        rms = torch.sqrt(torch.mean(audio_aug**2, dim=1, keepdim=True))
        noise_level = rms * self.noise_intensity
        
        audio_aug = audio_aug + noise * noise_level

        # ---------------------------------------------------
        # 3. (Time Domain Masking / Cutout)
        # ---------------------------------------------------
        
        mask_len = int(L * self.mask_ratio)
        start_indices = torch.randint(0, L - mask_len, (B,), device=device)
        indices = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        starts = start_indices.unsqueeze(1)
        ends = starts + mask_len
        mask = ~((indices >= starts) & (indices < ends))
        
        audio_aug = audio_aug * mask.float()

        return audio_aug
    