import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable
import math
class AOCloss(nn.Module):
    '''
    AOC loss function.
    based on the paper:
    Adaptive Centroid Shift Loss (AOCloss) method for Audio Deepfake Detection
    '''
    def __init__(self, embedding_dim=2):
        super(AOCloss, self).__init__()
        self.embedding_dim = embedding_dim
        self.centroid = None
        self.n = 0  # Total number of bonafide samples encountered

    def update_centroid(self, bonafide_embeddings):
        s = bonafide_embeddings.shape[0]
        if s == 0:
            return

        Ei = bonafide_embeddings.mean(dim=0).detach()  # Detach to avoid graph tracking

        if self.centroid is None:
            self.centroid = Ei
            self.n = s
        else:
            self.centroid = ((self.n * self.centroid.detach()) + (s * Ei)) / (self.n + s)
            self.n += s

    def one_class_loss(self, bonafide_embeddings, fake_embeddings):
        if self.centroid is None:
            raise ValueError("Centroid has not been initialized with bonafide samples.")

        # Normalize embeddings and centroid
        centroid_norm = F.normalize(self.centroid.detach(), p=2, dim=0)
        bonafide_norm = F.normalize(bonafide_embeddings, p=2, dim=1)
        fake_norm = F.normalize(fake_embeddings, p=2, dim=1)

        # Compute cosine similarity
        bonafide_similarity = torch.matmul(bonafide_norm, centroid_norm)
        fake_similarity = torch.matmul(fake_norm, centroid_norm)

        Mb = bonafide_embeddings.shape[0]
        Ms = fake_embeddings.shape[0]

        if Mb == 0 :
            loc = 1 + (torch.sum(fake_similarity) / Ms)
        elif Ms ==0:
            loc = 1 + (-torch.sum(bonafide_similarity) / Mb)
        else:
            loc = 1 + (-torch.sum(bonafide_similarity) / Mb) + (torch.sum(fake_similarity) / Ms)
        return loc

    def forward(self, embeddings, labels=None,stage = "train"):
        bonafide_embeddings = embeddings[labels == 1]
        fake_embeddings = embeddings[labels == 0]
        if stage =="train":
            self.update_centroid(bonafide_embeddings)
        loss = self.one_class_loss(bonafide_embeddings, fake_embeddings)
        return loss
