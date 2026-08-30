import torch
import torch.nn.functional as F
from models.OneClass.extractor import Extractor
from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification
from myutils.aocloss_cross_domain import AOCloss_CrossDomain
from myutils.shifter import Distribution_shifter

class CA_SOADD_cross_domain(DeepfakeAudioClassification):
    def __init__(self, loss_weight = [1.0,1.0], cfg=None, args=None, post= False,language_list = ["fr","it","pl","ru","uk"],**kwargs):
        super().__init__()
        
        self.cfg = cfg
        self.args = args
        
        # 1. backbone
        self.model = Extractor(post = post,cross_domain=True)
        self.shifter = Distribution_shifter()
        self.embed_dim = 128 
        self.language_list = language_list
        self.loss_fn = AOCloss_CrossDomain(embedding_dim=self.embed_dim,language_list=self.language_list,inter_center_weight = 10)
        self.loss_weight=loss_weight
        self.configure_normalizer()

        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6, weight_decay=1e-4)
        return optimizer

    def _shared_pred(self, batch, batch_idx, stage='train', **kwargs):
        audio = batch["audio"]
        labels = batch["label"]
        langs = batch["language"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        res = self.model(audio,labels)
        embedding = res["final_feat"]
        layer_w = res['layer_weights']
        if stage == 'train':
            audio_shifted = self.shifter(audio)
            feat_shifted = self.model(audio_shifted)["final_feat"]

        logits = []
        batch_size = embedding.shape[0]
        for i in range(batch_size):
            lang = langs[i]
            centroid = self.loss_fn.get_centroid_by_lang(lang)
            
            if centroid.sum() == 0:
                 logits.append(torch.tensor(0.0, device=embedding.device))
            else:
                emb_norm = F.normalize(embedding[i].unsqueeze(0), p=2, dim=1)
                cent_norm = F.normalize(centroid.unsqueeze(0).to(embedding.device), p=2, dim=1)
                
                sim = torch.matmul(emb_norm, cent_norm.T).squeeze()
                logits.append(sim)

        logit_tensor = torch.stack(logits)
        batch_pred = (logit_tensor > 0.95).int() 
        
        return {
            "final_feat": embedding,
            "feat_shifted": feat_shifted if self.training else None,
            "logit": logit_tensor,
            "pred": batch_pred, 
            "embedding": embedding
        }
    
    def calcuate_loss(self, batch_res, batch,stage):
        embeddings = batch_res["final_feat"]
        embedding_norm = F.normalize(embeddings, p=2, dim=1)
        
        if self.training:
            virtual_labels = torch.ones(embedding_norm.shape[0], device=embedding_norm.device, dtype=torch.long)
        else:
            input_label = batch["label"].type(torch.long)
            virtual_labels = input_label 
        langs_str = batch["language"]
        # OBJ cpt loss 
        loss_compact = self.loss_fn(embedding_norm, virtual_labels, langs_str, stage)
        # OBJ cabs loss
        loss_boundary = self.calculate_cabs_loss(batch_res["feat_shifted"]) 
        loss =  self.loss_weight[0] * loss_compact + self.loss_weight[1] * loss_boundary

        L_ent = self.model.layer_weighting.entropy_reg()
        loss = loss + 1e-2 * L_ent

        return loss
    
    def calculate_cabs_loss(self, feat_shifted):
        z_shifted = F.normalize(feat_shifted, p=2, dim=1) # [B, D]
        all_centroids = []
        for lang in self.language_list:
            c = getattr(self.loss_fn, f"centroid_{lang}")
            if c.sum() != 0:
                all_centroids.append(c)
        
        if len(all_centroids) == 0: return torch.tensor(0.0, device=feat_shifted.device)
        
        z_all_centers = F.normalize(torch.stack(all_centroids), p=2, dim=1)
        sim_matrix = torch.matmul(z_shifted, z_all_centers.T)
        return (sim_matrix.mean(dim=1)[0] + 1.0).mean()