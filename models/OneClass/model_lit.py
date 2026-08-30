import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.func import functional_call
import torchaudio
from tqdm import tqdm
from data.Data_Aug import SafeRawAugmentor
from models.OneClass.extractor import Extractor
from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification
from myutils.aocloss import AOCloss
from myutils.shifter import Distribution_shifter
class CA_SOADD_Lit(DeepfakeAudioClassification):
    def __init__(self, warmup_epoch=8, loss_weight=[5.0,1.0,1.0],cfg=None, args=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.model = Extractor()
        self.embed_dim = 128 
        
        self.register_buffer("centroid", torch.zeros(self.embed_dim))
        self.augmentor = SafeRawAugmentor(noise_intensity=0.1, mask_ratio=0.1)
        self.shifter = Distribution_shifter()
        self.configure_loss_fn()
        self.warmup_epoch = warmup_epoch
        self.loss_weight=loss_weight

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6, weight_decay=1e-4)
        return optimizer
    
    def configure_loss_fn(self):
        self.loss_fn = AOCloss(embedding_dim=self.embed_dim)
        
        if torch.any(self.centroid != 0):
            self.loss_fn.centroid = self.centroid.clone()

    def _shared_pred(self, batch, batch_idx, stage='train', **kwargs):
        audio = batch["audio"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        # OBJ_cpt
        res = self.model(audio)
        feat_clean = res["final_feat"]
        if self.warmup_epoch != -1:
            # OBJ_binv
            audio_noisy_input = self.augmentor(audio)
            feat_noisy = self.model(audio_noisy_input)["final_feat"]

            # OBJ_cabs
            audio_shifted = self.shifter(audio)
            feat_shifted = self.model(audio_shifted)["final_feat"]
        
        if self.loss_fn.centroid is None:
            self.loss_fn.centroid = self.centroid
        centroid = self.loss_fn.centroid
        feat_norm = F.normalize(feat_clean, p=2, dim=1)
        centroid_norm = F.normalize(centroid.to(feat_clean.device), p=2, dim=0)
        similarity = torch.matmul(feat_norm, centroid_norm)
        logit = similarity
        
        batch_pred = (similarity > 0.95).int()

        return {
            "final_feat": feat_clean,
            "feat_noisy": feat_noisy,
            "feat_shifted": feat_shifted,
            "logit": logit,
            "pred": batch_pred
        }
        
    def calcuate_loss(self, batch_res, batch, stage):
        z_clean = F.normalize(batch_res["final_feat"], p=2, dim=1)
        if self.warmup_epoch != -1:
            z_noisy = F.normalize(batch_res["feat_noisy"], p=2, dim=1)
            z_shifted = F.normalize(batch_res["feat_shifted"], p=2, dim=1)

        if self.training:
            virtual_labels = torch.ones(z_clean.shape[0], device=z_clean.device, dtype=torch.long)
        else:
            virtual_labels = batch["label"].type(torch.long)
        
        # 1.OBJ cpt loss
        loss_compact = self.loss_fn(z_clean, virtual_labels, stage)        
        if self.current_epoch < self.warmup_epoch:
            loss = loss_compact
            return loss
        
        # 2.OBJ binv loss
        if self.warmup_epoch != -1:
            sim_consistency = (z_clean * z_noisy.detach()).sum(dim=1)
            loss_consistency = (1 - sim_consistency).mean()
            
            # 3.OBJ cabs loss
            centroid_norm = F.normalize(self.loss_fn.centroid.detach(), p=2, dim=0)
            sim_shifted = torch.matmul(z_shifted, centroid_norm)
            loss_boundary = torch.clamp(sim_shifted + 0.1, min=0.0).mean() 

            loss = self.loss_weight[0] * loss_compact 
            + self.loss_weight[1] * loss_consistency 
            + self.loss_weight[2]* loss_boundary
        else:
            loss = self.loss_weight[0] * loss_compact 
        L_ent = self.model.layer_weighting.entropy_reg()
        loss = loss + 1e-2 * L_ent
        return loss