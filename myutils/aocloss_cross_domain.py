import torch
import torch.nn as nn
import torch.nn.functional as F

class AOCloss_CrossDomain(nn.Module):
    """
    Multi-language version of AOC Loss.
    Maintains independent Bonafide Centroids for each language to handle cross-domain shifts.
    """
    def __init__(self, embedding_dim=128, language_list=["fr", "it", "pl", "ru", "uk"], inter_center_weight=1):
        super(AOCloss_CrossDomain, self).__init__()
        self.embedding_dim = embedding_dim
        self.language_list = language_list
        self.lang2id = {lang: i for i, lang in enumerate(language_list)}
        self.inter_center_weight = inter_center_weight

        # Dynamically register buffers for centroids and counters
        for lang in language_list:
            self.register_buffer(f"centroid_{lang}", torch.zeros(embedding_dim))
            self.register_buffer(f"n_{lang}", torch.tensor(0, dtype=torch.long))

    def calculate_inter_center_loss(self):
        """
        Calculates squared loss between centroids to encourage alignment.
        """
        centroids = [getattr(self, f"centroid_{lang}") for lang in self.language_list]
        centroids_tensor = torch.stack(centroids)
        
        # Filter valid (initialized) centroids
        valid_mask = (centroids_tensor.sum(dim=1) != 0)
        valid_centroids = centroids_tensor[valid_mask]
        
        num_valid = valid_centroids.shape[0]
        if num_valid < 2:
            return torch.tensor(0.0, device=centroids_tensor.device)
            
        valid_centroids_norm = F.normalize(valid_centroids, p=2, dim=1)
        
        # Compute pairwise similarity matrix
        sim_matrix = torch.matmul(valid_centroids_norm, valid_centroids_norm.T)
        rows, cols = torch.triu_indices(num_valid, num_valid, offset=1)
        off_diag_sims = sim_matrix[rows, cols]
        
        # Squared error loss: target similarity is 1.0
        inter_loss = torch.mean((1 - off_diag_sims) ** 2)
        return inter_loss

    def update_centroid(self, embeddings, labels, lang_indices):
        """
        Updates language-specific centroids using moving average.
        """
        bonafide_mask = (labels == 1)
        if not bonafide_mask.any():
            return

        present_langs = torch.unique(lang_indices[bonafide_mask])
        for lid in present_langs:
            lang_name = self.language_list[lid.item()]
            mask = (lang_indices == lid) & bonafide_mask
            if not mask.any():
                continue
                
            features = embeddings[mask]
            s = features.shape[0]
            Ei = features.mean(dim=0).detach()
            
            centroid_name = f"centroid_{lang_name}"
            n_name = f"n_{lang_name}"
            current_centroid = getattr(self, centroid_name)
            current_n = getattr(self, n_name)
            
            # Cumulative moving average update
            if current_n == 0:
                new_centroid = Ei
                new_n = s
            else:
                new_centroid = ((current_n * current_centroid) + (s * Ei)) / (current_n + s)
                new_n = current_n + s
            
            setattr(self, centroid_name, new_centroid)
            setattr(self, n_name, torch.tensor(new_n, device=current_n.device))

    def forward(self, embeddings, labels, langs_str, stage="train"):
        """
        Calculates the AOC Loss combined with inter-center regularization.
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        lang_indices = torch.tensor(
            [self.lang2id[l] for l in langs_str], 
            device=device, 
            dtype=torch.long
        )
        
        if self.training or stage == "train":
            self.update_centroid(embeddings, labels, lang_indices)
        
        batch_centroids = []
        valid_mask = []
        
        for lang_name in langs_str:
            c = getattr(self, f"centroid_{lang_name}")
            batch_centroids.append(c)
            valid_mask.append(c.sum() != 0)
            
        batch_centroids = torch.stack(batch_centroids).to(device)
        valid_mask = torch.tensor(valid_mask, device=device, dtype=torch.bool)
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        emb_norm = F.normalize(embeddings, p=2, dim=1)
        center_norm = F.normalize(batch_centroids, p=2, dim=1)
        
        # Cosine similarity for current batch against respective language centroids
        similarity = torch.sum(emb_norm * center_norm, dim=1)

        is_bonafide = (labels == 1)
        is_spoof = (labels == 0)
        
        loss_bonafide = 0.0
        loss_spoof = 0.0
        
        # Maximize similarity for Bonafide
        bona_indices = is_bonafide & valid_mask
        mb = bona_indices.sum()
        if mb > 0:
            loss_bonafide = -torch.sum(similarity[bona_indices]) / mb
            
        # Minimize similarity for Spoof
        spoof_indices = is_spoof & valid_mask
        ms = spoof_indices.sum()
        if ms > 0:
            loss_spoof = torch.sum(similarity[spoof_indices]) / ms
            
        base_loss = 1.0 + loss_bonafide + loss_spoof
        inter_loss = self.calculate_inter_center_loss()
        
        # Apply inter-center regularization if diversity is high
        if inter_loss < 0.05:
            return base_loss + (self.inter_center_weight * inter_loss)
        return base_loss

    def get_centroid_by_lang(self, lang_str):
        return getattr(self, f"centroid_{lang_str}")