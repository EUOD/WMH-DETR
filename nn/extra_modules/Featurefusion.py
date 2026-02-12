import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.conv import Conv
class DynamicFeatureFusion(nn.Module):
    """动态特征融合 - 学习最优融合权重"""
    def __init__(self, channels):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, f1, f2):
        # f1: 上采样特征, f2: 横向连接特征
        cat_feat = torch.cat([f1, f2], dim=1)
        weights = self.weight_net(cat_feat)
        w1, w2 = weights[:, 0:1], weights[:, 1:2]
        return f1 * w1 + f2 * w2