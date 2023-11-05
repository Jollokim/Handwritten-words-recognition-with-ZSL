import torch

import torch.nn as nn
from torch.nn import functional as F

from timm.models.registry import register_model

__all__ = [
    'standard_loss',
    'PHOSCCosine_loss',
    'Cosine_loss'
]


class PHOSCLoss(nn.Module):
    def __init__(self, phos_w=4.5, phoc_w=1):
        super().__init__()

        self.phos_w = phos_w
        self.phoc_w = phoc_w

    def forward(self, y: dict, targets: torch.Tensor):
        phos_loss = self.phos_w * F.mse_loss(y['phos'], targets['phos'])
        phoc_loss = self.phoc_w * F.cross_entropy(y['phoc'], targets['phoc'])

        loss = phos_loss + phoc_loss
        return loss
    
class PHOSCCosineLoss(nn.Module):
    def __init__(self, phos_w=4.5, phoc_w=1, cos_w=1):
        super().__init__()

        self.phos_w = phos_w
        self.phoc_w = phoc_w
        self.cos_w = cos_w

        self.cosine_criterion = nn.CosineEmbeddingLoss()

    def forward(self, y: dict, targets: dict):
        phos_loss = self.phos_w * F.mse_loss(y['phos'], targets['phos'])
        phoc_loss = self.phoc_w * F.cross_entropy(y['phoc'], targets['phoc'])

        phosc_out = torch.cat((y['phoc'], y['phos']), dim=1)
        phosc_target = targets['phosc']
        similarities = targets['sim']

        cosine_loss = self.cos_w * F.cosine_embedding_loss(phosc_out, phosc_target, similarities)

        loss = phos_loss + phoc_loss + cosine_loss
        return loss


class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.cosine_criterion = nn.CosineEmbeddingLoss()

    def forward(self, y: dict, targets: dict):

        phosc_out = torch.cat((y['phoc'], y['phos']), dim=1)
        phosc_target = targets['phosc']
        similarities = targets['sim']

        cosine_loss = F.cosine_embedding_loss(phosc_out, phosc_target, similarities) * 100

        return cosine_loss
    

@register_model
def standard_loss(**kwargs):
    return PHOSCLoss(phos_w=kwargs['phos_w'], phoc_w=kwargs['phoc_w'])

@register_model
def PHOSCCosine_loss(**kwargs):
    return PHOSCCosineLoss(phos_w=kwargs['phos_w'], phoc_w=kwargs['phoc_w'], cos_w=kwargs['cosine_w'])

@register_model
def Cosine_loss(**kwargs):
    return CosineLoss()


if __name__ == '__main__':

    y = {
        'phos': torch.randn((64, 165)),
        'phoc': torch.randn((64, 604))
    }
    target = {
        'phos': torch.randn((64, 165)),
        'phoc': torch.randn((64, 604)),
        'phosc': torch.randn((64, 769)),
        'sim': torch.ones((64))
    }

    print(y['phos'].shape, y['phoc'].shape)
    print(target['phos'].shape, target['phoc'].shape)

    criterion = PHOSCCosineLoss()

    loss = criterion(y, target)
    print(loss)