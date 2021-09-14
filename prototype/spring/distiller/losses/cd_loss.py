import torch
import torch.nn as nn


class CDLoss(nn.Module):
    """
    Channel Distillation: Channel-Wise Attention for Knowledge Distillation
    https://arxiv.org/abs/2006.01683
    """
    def __init__(self):
        super(CDLoss, self).__init__()

    def forward(self, s_features, t_features, **kwargs):
        loss = 0
        for s, t in zip(s_features, t_features):
            s = s.mean(dim=(2, 3), keepdim=False)
            t = t.mean(dim=(2, 3), keepdim=False)
            loss += torch.mean(torch.pow(s - t, 2))
        return loss
