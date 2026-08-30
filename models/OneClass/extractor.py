import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForPreTraining

class DynamicLayerWeighting(nn.Module):
    """
    Learns attention-based weights for multiple hidden layers to fuse them into a single representation.
    Based on the XLS-R layer selection paradigm.
    """
    def __init__(self, input_dim=1024, num_layers=25):
        super(DynamicLayerWeighting, self).__init__()
        self.fc_weight = nn.Linear(input_dim, 1) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden_states):
        # hidden_states: tuple of tensors, len L. Each tensor: (B, T, D)
        full_features = torch.stack(hidden_states, dim=1)  # (B, L, T, D)
        
        # Global average pooling over time dimension to represent each layer
        layer_reps = torch.mean(full_features, dim=2)  # (B, L, D)
        
        # Calculate raw scores and normalize to weights
        raw_scores = self.fc_weight(layer_reps)  # (B, L, 1)
        layer_weights = self.softmax(raw_scores).unsqueeze(-1)  # (B, L, 1, 1)
        
        fused_feature = torch.sum(full_features * layer_weights, dim=1)  # (B, T, D)
        return fused_feature, layer_weights.squeeze()

class TemporalAttentionRefinement(nn.Module):
    """
    Enhances temporal features using a bottleneck attention mechanism.
    Highlights salient frames (e.g., active speech) and suppresses noise/silence.
    """
    def __init__(self, input_dim=1024, attention_channels=128):
        super(TemporalAttentionRefinement, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, attention_channels, kernel_size=1),
            nn.Tanh(), 
            nn.Conv1d(attention_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, T, D) -> Transpose to (B, D, T) for Conv1d
        x_in = x.transpose(1, 2)
        
        attn_scores = self.attention(x_in)  # (B, 1, T)
        attn_weights = F.softmax(attn_scores, dim=2)  # (B, 1, T)
        
        # Element-wise weighting to refine temporal features
        refined_feat = x_in * attn_weights
        return refined_feat.transpose(1, 2), attn_weights

class StableLayerSelection(nn.Module):
    """
    Implements a stable, learnable softmax-based layer weighting mechanism.
    """
    def __init__(self, num_layers=24, temperature=2.0, init="uniform"):
        super().__init__()
        self.num_layers = num_layers
        self.temperature = temperature
        logits = torch.zeros(num_layers)
        if init == "uniform":
            logits.zero_()
        self.layer_logits = nn.Parameter(logits)

    def forward(self, hidden_states):
        full = torch.stack(hidden_states, dim=1)  # (B, L, T, D)
        w = F.softmax(self.layer_logits / self.temperature, dim=0)  # (L,)
        fused = (full * w.view(1, -1, 1, 1)).sum(dim=1)  # (B, T, D)
        return fused, w

    def entropy_reg(self):
        """Entropy regularization to control layer sparsity/diversity."""
        w = F.softmax(self.layer_logits / self.temperature, dim=0)
        ent = -(w * (w + 1e-12).log()).sum()
        return -ent

class LangStableLayerSelection(nn.Module):
    """
    Language-aware layer selection. Maintains independent logits for different languages
    to adapt the feature fusion to language-specific acoustic properties.
    """
    def __init__(self, num_layers=24, temperature=2.0, init="uniform", lang2id=None):
        super().__init__()
        self.num_layers = num_layers
        self.temperature = temperature
        
        if lang2id is None:
            lang2id = {"en": 0, "de": 1, "es": 2, "fr": 3, "it": 4, "pl": 5, "ru": 6, "uk": 7}
        self.lang2id = lang2id
        self.num_langs = len(lang2id)

        logits = torch.zeros(self.num_langs, num_layers)
        self.layer_logits = nn.Parameter(logits)

    def forward(self, hidden_states, langs=None):
        full = torch.stack(hidden_states, dim=1)
        B, L, T, D = full.shape

        if langs is None:
            # Fallback to global average weight
            logits = self.layer_logits.mean(dim=0)
            w = F.softmax(logits / self.temperature, dim=0)
            fused = (full * w.view(1, -1, 1, 1)).sum(dim=1)
            return fused, w

        # Map language strings to IDs and select corresponding logits
        if isinstance(langs, (list, tuple)):
            lang_ids = torch.tensor([self.lang2id[s] for s in langs], device=full.device)
        elif torch.is_tensor(langs):
            lang_ids = langs.to(full.device).long().view(-1)
        else:
            lang_ids = torch.full((B,), self.lang2id[langs], device=full.device, dtype=torch.long)

        logits = self.layer_logits.index_select(0, lang_ids)  # (B, L)
        w = F.softmax(logits / self.temperature, dim=1)
        fused = (full * w.view(B, L, 1, 1)).sum(dim=1)
        return fused, w

class RunningStandardize(nn.Module):
    """
    Implements Online/Running Standardization using Welford's algorithm.
    Ensures feature normalization based on bonafide sample statistics.
    """
    def __init__(self, dim, eps=1e-5, use_var=True):
        super().__init__()
        self.eps = eps
        self.use_var = use_var
        self.register_buffer("n", torch.tensor(0, dtype=torch.long))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("M2", torch.zeros(dim))

    @torch.no_grad()
    def update(self, x):
        if x.numel() == 0: return
        b = x.size(0)
        batch_mean = x.mean(dim=0)
        batch_M2 = ((x - batch_mean) ** 2).sum(dim=0) if self.use_var else torch.zeros_like(batch_mean)

        n0 = int(self.n.item())
        n1 = n0 + b
        if n0 == 0:
            self.mean.copy_(batch_mean)
            self.M2.copy_(batch_M2)
        else:
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * (b / n1)
            if self.use_var:
                new_M2 = self.M2 + batch_M2 + (delta ** 2) * (n0 * b / n1)
                self.M2.copy_(new_M2)
            self.mean.copy_(new_mean)
        self.n.fill_(n1)

    def forward(self, x, label, update_stats: bool):
        if update_stats:
            x_update = x[label == 1] # Update stats using only bonafide samples
            self.update(x_update)
        
        x = x - self.mean.unsqueeze(0)
        if self.use_var and int(self.n.item()) > 1:
            var = self.M2 / (self.n.float() - 1.0)
            x = x / torch.sqrt(var.unsqueeze(0) + self.eps)
        return x

class Extractor(nn.Module):
    """
    Comprehensive feature extractor.
    Pipeline: Frozen XLS-R -> Adaptive Layer Fusion -> Temporal Attention -> Projection.
    """
    def __init__(self, post=False, cross_domain=False):
        super(Extractor, self).__init__()
        self.xlsr = AutoModelForPreTraining.from_pretrained("path/to/wav2vec2-xls-r-300m")
        input_dim = 1024

        if cross_domain:
            self.layer_weighting = LangStableLayerSelection()
        else:
            self.layer_weighting = StableLayerSelection()
        
        self.temporal_refiner = TemporalAttentionRefinement(input_dim=input_dim)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, 128)
        )
        if post:
            self.post_std = RunningStandardize(dim=128)

    def forward(self, x, labels=None, langs=None):
        if x.dim() == 3:
            x = x.squeeze(1)
            
        outputs = self.xlsr(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:] 
        
        # Layer Fusion
        fused_feat, layer_w = self.layer_weighting(hidden_states, langs=langs)
        
        # Temporal Attention and Global Aggregation
        refined_feat, _ = self.temporal_refiner(fused_feat)
        representation = refined_feat.sum(dim=1)

        # Dimension Projection
        projected_feat = self.projector(representation)

        # Running Standardization
        if hasattr(self, 'post_std'):
            final_feat = self.post_std(projected_feat, labels, update_stats=self.training)
            return {
                'pre_feat': projected_feat,
                'final_feat': final_feat,
                'layer_weights': layer_w
            }
            
        return {
            'final_feat': projected_feat,
            'layer_weights': layer_w
        }