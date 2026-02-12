import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
   
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops   

import torch     
import torch.nn as nn  
import torch.nn.functional as F 
from functools import partial 
import pywt
from einops import rearrange 
from timm.layers import to_2tuple
try: 
    from kat_rational import KAT_Group
except ImportError as e:
    print(f'from kat_rational import KAT_Group Failure. message:{e}')
    
class CBS(nn.Module):
    """Conv + BatchNorm + SiLU"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class KAN(nn.Module): 
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """   
    def __init__(
            self,
            in_features,   
            hidden_features=None,
            out_features=None,   
            act_layer=None,   
            norm_layer=None,    
            bias=True,   
            drop=0.,
            use_conv=False,
            act_init="gelu",
    ):   
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features    
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0]) 
        self.act1 = KAT_Group(mode="identity")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()  
        self.act2 = KAT_Group(mode=act_init)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward_kan(self, x):    
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)  
        x = self.fc2(x)
        return x
  
    def forward(self, x):  
        B, C, H, W = x.size()    
        x_nlc = x.flatten(2).permute(0, 2, 1)
        x_nlc = self.forward_kan(x_nlc)
        x_nchw = x_nlc.permute(0, 2, 1).view([B, -1, H, W]).contiguous()
        return x_nchw

class SKA(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(SKA, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
class SKATTransformerBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        mlp_ratio=4., 
        drop=0., 
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_conv=False,
        bias=False
    ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = SKA(dim=dim, num_heads=num_heads, bias=bias)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dropout1 = nn.Dropout(drop)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = KAN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            drop=drop,
            use_conv=use_conv,
            act_init="gelu"
        )
        
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # First block
        x_norm1 = x.flatten(2).permute(0, 2, 1)
        x_norm1 = self.norm1(x_norm1)
        x_norm1 = x_norm1.permute(0, 2, 1).view(B, C, H, W)
        
        attn_out = self.attn(x_norm1)
        attn_out = self.dropout1(attn_out)
        x = x + self.drop_path1(attn_out)
        
        # Second block
        x_norm2 = x.flatten(2).permute(0, 2, 1)
        x_norm2 = self.norm2(x_norm2)
        x_norm2 = x_norm2.permute(0, 2, 1).view(B, C, H, W)
        
        mlp_out = self.mlp(x_norm2)
        mlp_out = self.dropout2(mlp_out)
        x = x + self.drop_path2(mlp_out)
        
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

def conv_relu_bn(in_channel, out_channel, dirate=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, 
                  stride=1, padding=dirate, dilation=dirate),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )     
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_dim = max(1, in_planes // ratio)

        self.fc1 = nn.Conv2d(in_planes, hidden_dim, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_dim, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        res = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * res

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x_source = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * x_source      

class WaveletDualAttention(nn.Module):
    """
    小波域双注意力模块
    流程: Input → DWT → Spatial Attention → Channel Attention → IDWT → Output
    """
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(WaveletDualAttention, self).__init__()
        
        # 小波域特征融合 (4个子带 → 1个特征图)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 空间注意力模块
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)
        
        # 通道注意力模块
        self.channel_attention = ChannelAttention(channels, ratio=ratio)
        
        # 投影回小波域 (1个特征图 → 4个子带)
        self.projection = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        identity = x  # 保存输入用于残差连接
        
        # Step 1: DWT - 小波变换分解
        ll, lh, hl, hh = dwt(x)
        
        # Step 2: 拼接四个子带 [B, C*4, H/2, W/2]
        wavelet_features = torch.cat([ll, lh, hl, hh], dim=1)
        
        # Step 3: 融合到统一特征空间 [B, C, H/2, W/2]
        fused = self.fusion(wavelet_features)
        
        # Step 4: 空间注意力
        spatial_out = self.spatial_attention(fused)
        
        # Step 5: 通道注意力
        channel_out = self.channel_attention(spatial_out)
        
        # Step 6: 投影回小波域 [B, C*4, H/2, W/2]
        projected = self.projection(channel_out)
        
        # Step 7: 分离四个子带
        B, C4, H, W = projected.shape
        C = C4 // 4
        ll_out = projected[:, :C, :, :]
        lh_out = projected[:, C:2*C, :, :]
        hl_out = projected[:, 2*C:3*C, :, :]
        hh_out = projected[:, 3*C:, :, :]
        
        # Step 8: IDWT - 逆小波变换重建
        reconstructed = idwt(ll_out, lh_out, hl_out, hh_out)
        
        # Step 9: 残差连接
        output = reconstructed + identity
        
        return output
                                                                                                   
def dwt(x):
    """
    使用 PyWavelets 实现二维离散小波变换
    输入: x -> [B, C, H, W]
    输出: LL, LH, HL, HH -> [B, C, H//2, W//2]                                                                                                    
    """
    B, C, H, W = x.shape
    LL, LH, HL, HH = [], [], [], []
    for b in range(B):
        ll_, lh_, hl_, hh_ = [], [], [], []
        for c in range(C):
            coeffs2 = pywt.dwt2(x[b, c].detach().cpu().numpy(), 'haar')                                                                                                     
            ll, (lh, hl, hh) = coeffs2
            ll_.append(torch.tensor(ll))
            lh_.append(torch.tensor(lh))
            hl_.append(torch.tensor(hl))
            hh_.append(torch.tensor(hh))
        LL.append(torch.stack(ll_))
        LH.append(torch.stack(lh_))
        HL.append(torch.stack(hl_))
        HH.append(torch.stack(hh_))
    return (torch.stack(LL).to(x.device),
            torch.stack(LH).to(x.device),
            torch.stack(HL).to(x.device),
            torch.stack(HH).to(x.device))

def idwt(ll, lh, hl, hh):
    """
    使用 PyWavelets 实现二维离散小波逆变换
    输入: LL, LH, HL, HH -> [B, C, H, W]
    输出: 重建图像 [B, C, H*2, W*2]
    """
    B, C, H, W = ll.shape
    out = []
    for b in range(B):
        rec = []
        for c in range(C):
            coeffs2 = (
                ll[b, c].detach().cpu().numpy(),
                (
                    lh[b, c].detach().cpu().numpy(),                                                                                                     
                    hl[b, c].detach().cpu().numpy(),                                                                                                     
                    hh[b, c].detach().cpu().numpy()                                                                                                     
                )
            )
            rec_ = pywt.idwt2(coeffs2, 'haar')                                                                                                     
            rec.append(torch.tensor(rec_))
        out.append(torch.stack(rec))
    return torch.stack(out).to(ll.device)

class DualBranchFusionBlock(nn.Module):
    """
    双分支融合模块（5路Concat）
    
    Concat输入来源：
    1. 左分支 CBS_1 输出 (C/4)
    2. 左分支 CBS_2 输出 (C/4)
    3. Split 直连 (C/2)
    4. HWFE 直连 (C/2)
    5. 右分支 Transformer 输出 (C/2)
    
    Args:
        c1: 输入通道数
        c2: 输出通道数 (默认等于c1)
        num_heads: Transformer头数 (默认8)
        mlp_ratio: MLP扩展比例 (默认4.0)
    """
    def __init__(self, c1, c2=None, num_heads=8, mlp_ratio=4.0):
        super(DualBranchFusionBlock, self).__init__()
        
        if c2 is None:
            c2 = c1
        
        self.c1 = c1
        self.c2 = c2
        
        # 输入CBS
        self.input_cbs = CBS(c1, c1, kernel_size=3, stride=1, padding=1)
        
        # 分支通道数
        self.branch_channels = c1 // 2
        
        # ========== 左分支 ==========
        # HWFE (WaveletDualAttention) 替代 UEMA
        self.left_hwfe = WaveletDualAttention(
            channels=self.branch_channels,
            ratio=16,
            kernel_size=7
        )
        
        # CBS after HWFE
        self.left_cbs1 = CBS(self.branch_channels, self.branch_channels, kernel_size=3, stride=1, padding=1)
        
        # Split后的两个分支通道数
        self.left_split_channels = self.branch_channels // 4
        
        # Split后的两个CBS
        self.left_cbs2_1 = CBS(self.left_split_channels, self.left_split_channels, kernel_size=3, stride=1, padding=1)
        self.left_cbs2_2 = CBS(self.left_split_channels, self.left_split_channels, kernel_size=3, stride=1, padding=1)
        
        # ========== 右分支 ==========
        # SKATTransformerBlock 替代 LEFM + LayerNorm + SS2D
        self.right_transformer = SKATTransformerBlock(
            dim=self.branch_channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=0.0,
            drop_path=0.1,
            bias=False
        )
        
        # ========== 最终融合 ==========
        # 5路Concat: 2个CBS(C/4) + Split直连(C/2) + HWFE直连(C/2) + Transformer(C/2)
        # 总通道数: C/4 + C/4 + C/2 + C/2 + C/2 = 2C
        concat_channels = self.left_split_channels * 2 + self.branch_channels * 3
        
        # Concat后的输出CBS
        self.output_cbs = CBS(concat_channels, c2, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            out: 输出特征 [B, C2, H, W]
        """
        # 输入CBS
        x = self.input_cbs(x)  # [B, C, H, W]
        
        # Split成两个分支
        B, C, H, W = x.shape
        left_x = x[:, :self.branch_channels, :, :]   # [B, C/2, H, W]
        right_x = x[:, self.branch_channels:, :, :]  # [B, C/2, H, W]
        
        # ========== 左分支处理 ==========
        # HWFE (替代UEMA)
        hwfe_out = self.left_hwfe(left_x)  # [B, C/2, H, W] - 用于直连
        
        # CBS
        left_out = self.left_cbs1(hwfe_out)  # [B, C/2, H, W]
        
        # Split成两个小分支（这里也保存split的结果用于直连）
        split_out = left_out  # [B, C/2, H, W] - 用于直连
        
        left_split1 = left_out[:, :self.left_split_channels, :, :]  # [B, C/4, H, W]
        left_split2 = left_out[:, self.left_split_channels:, :, :]  # [B, C/4, H, W]
        
        # 两个CBS
        cbs_out1 = self.left_cbs2_1(left_split1)  # [B, C/4, H, W]
        cbs_out2 = self.left_cbs2_2(left_split2)  # [B, C/4, H, W]
        
        # ========== 右分支处理 ==========
        # SKATTransformerBlock (替代LEFM+LayerNorm+SS2D)
        right_out = self.right_transformer(right_x)  # [B, C/2, H, W]
        
        # ========== 5路融合 ==========
        # 1. 左分支 CBS_1 输出 (C/4)
        # 2. 左分支 CBS_2 输出 (C/4)
        # 3. Split 直连 (C/2)
        # 4. HWFE 直连 (C/2)
        # 5. 右分支 Transformer 输出 (C/2)
        out = torch.cat([
            cbs_out1,      # [B, C/4, H, W]
            cbs_out2,      # [B, C/4, H, W]
            split_out,     # [B, C/2, H, W] - Split直连
            hwfe_out,      # [B, C/2, H, W] - HWFE直连
            right_out      # [B, C/2, H, W] - 右分支
        ], dim=1)  # [B, 2C, H, W]
        
        # 输出CBS
        out = self.output_cbs(out)  # [B, C2, H, W]
        
        return out
