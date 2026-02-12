import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
   
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops   

import torch     
import torch.nn as nn  
import math
import torch.nn.functional as F 
from functools import partial 
from ultralytics.nn.modules.conv import Conv
import pywt
from einops import rearrange 
from timm.layers import to_2tuple
try: 
    from kat_rational import KAT_Group
except ImportError as e:
    print(f'from kat_rational import KAT_Group Failure. message:{e}')


# --------------------------------------------------------  HSG_X_TKSABlock   ------------------------------------------------------------
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

class TKSABlock(nn.Module):
    def __init__(self, c1, c2=None, alpha=0.5, reduction=8, num_heads=8):
        super(TKSABlock, self).__init__()
        
        if c2 is None:
            c2 = c1
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        
        self.split_channels = int(c1 * alpha)
        self.gate_channels = c1 - self.split_channels
        self.reduced_channels = max(4, self.gate_channels // reduction)
        
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(self.gate_channels, self.reduced_channels, 1),
            nn.BatchNorm2d(self.reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        self.gct_stream = GctStream(self.reduced_channels)
        self.rel_stream = RelStream(self.reduced_channels)
        
        # 使用完整的SKATTransformerBlock
        self.SKAT_stream = SKATTransformerBlock(
            dim=self.reduced_channels,
            num_heads=num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
            bias=False
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.reduced_channels * 2, self.reduced_channels, 1),
            nn.BatchNorm2d(self.reduced_channels),
            nn.SiLU()
        )
        
        # 扩展卷积（调整维度）
        self.expand_conv = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.gate_channels, 1),
            nn.BatchNorm2d(self.gate_channels)
        )
        # 最终输出卷积
        concat_channels = self.split_channels + 2 * self.gate_channels
        
        self.final_conv = nn.Conv2d(concat_channels, c2, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # ===== 步骤1: 通道分割 =====
        identity_feat = x[:, :self.split_channels, :, :]
        gate_feat = x[:, self.split_channels:, :, :]
        # ===== 步骤2: 降维 =====
        reduced_feat = self.reduction_conv(gate_feat)
        # ===== 步骤3: 三个并行流 =====
        gct_out = self.gct_stream(reduced_feat)
        rel_out = self.rel_stream(reduced_feat)
        cons_out = self.SKAT_stream(reduced_feat)  # SKATTransformerBlock

        # ===== 步骤4：Rel和Cons拼接 =====
        rel_cons = torch.cat([rel_out, cons_out], dim=1)
        
        # ===== 步骤5：融合卷积 + SiLU =====
        rel_cons_fused = self.fusion_conv(rel_cons)
        
        # ===== 步骤6：与Gct相乘 =====
        interaction = gct_out * rel_cons_fused
        
        # ===== 步骤7：扩展维度 =====
        interaction_expanded = self.expand_conv(interaction)
        
        # ===== 步骤8：残差连接 =====
        enhanced_feat = gate_feat + interaction_expanded
        
        # ⚠️ ===== 步骤9：3分支拼接 =====
        # 拼接: [identity, enhanced, gate]
        # 通道数: split_channels + gate_channels + gate_channels
        out = torch.cat([identity_feat, enhanced_feat, gate_feat], dim=1)
        
        # ===== 步骤10：最终卷积 =====
        out = self.final_conv(out)
        
        return out
# -----------------------------------------------------------------------------------------------------------------------

#  --------------             HSG_X_HWFEBlock           ---------------------------------------------------------------------  
def build_wavelet_kernels(device=None, dtype=torch.float32):                                                                                                                                                                     # 微信公众号:AI缝合术
    """
    返回 2x2 的四个 2D 分析核：LL, LH, HL, HH
    对于 Db1 (Haar)：h0=[1/sqrt2, 1/sqrt2], h1=[-1/sqrt2, 1/sqrt2]                                                                                                                                                                     # 微信公众号:AI缝合术
    2D 核是外积：h_row^T * h_col
    """
    s = 1.0 / math.sqrt(2.0)
    h0 = torch.tensor([s, s], dtype=dtype, device=device)      # 低通                                                                                                                                                                     # 微信公众号:AI缝合术
    h1 = torch.tensor([-s, s], dtype=dtype, device=device)     # 高频                                                                                                                                                                     # 微信公众号:AI缝合术
    # 外积得到 2x2 核
    LL = torch.ger(h0, h0)  # 低-低
    LH = torch.ger(h0, h1)  # 低-高（垂直边）
    HL = torch.ger(h1, h0)  # 高-低（水平边）
    HH = torch.ger(h1, h1)  # 高-高（对角）
    # 形状统一为 (1,1,2,2) 方便后续扩展到 groups=C
    filt = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)                                                                                                                                                                     # 微信公众号:AI缝合术
    return filt  # (4,1,2,2)


# =========================
# Wavelet Attention 模块
# =========================
class WaveletAttention(nn.Module):
    """
    实现步骤：
    X --DWT--> (LH, HL, HH, LL)
         高频阈值化 -> concat -> 1x1 conv 融合 -> 与 LL 做 IDWT -> X_re
         GAP -> (可选FC) -> Softmax -> 通道权重
         输出： Final = weight * X
    """
    def __init__(self, channels, use_fc=True):
        super().__init__()
        self.channels = channels
        self.use_fc = use_fc

        # 软阈值参数（3 个高频子带 * C），sigmoid 约束到 0~1，再乘以 mean(|x|)
        self.theta = nn.Parameter(torch.zeros(3, channels, 1, 1))

        # 高频子带融合：将 3C -> C
        self.fuse = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False)                                                                                                                                                                     # 微信公众号:AI缝合术

        # GAP 后可选的 FC（保持维度 C->C）
        if use_fc:
            self.fc = nn.Linear(channels, channels, bias=True)

        # 小波核（注册为 buffer，参与 to(device) 但不训练）
        filt = build_wavelet_kernels()
        self.register_buffer("w_analysis", filt)   # (4,1,2,2)
        self.register_buffer("w_synthesis", filt)  # Db1 正交：合成=分析

    # ---------- DWT 与 IDWT ----------
    def dwt(self, x):
        """
        x: (B,C,H,W)
        返回：LH, HL, HH, LL 以及中间 size 信息
        """
        B, C, H, W = x.shape

        # 零填充到偶数尺寸，避免边界丢失
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)                                                                                                                                                                     # 微信公众号:AI缝合术

        # 组卷积：每个通道使用同一组 4 个滤波器
        # 权重形状需要扩展为 (4*C, 1, 2, 2) 并 groups=C
        weight = self.w_analysis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        y = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)  # (B,4C,H/2,W/2)                                                                                                                                                                     # 微信公众号:AI缝合术

        # 按子带拆分
        y = y.view(B, C, 4, y.size(-2), y.size(-1)).contiguous()                                                                                                                                                                     # 微信公众号:AI缝合术
        LL = y[:, :, 0]  # (B,C,h,w)
        LH = y[:, :, 1]
        HL = y[:, :, 2]
        HH = y[:, :, 3]
        return LH, HL, HH, LL

    def idwt(self, LH, HL, HH, LL):
        """
        逆变换：将四个子带重建为 (B,C,H,W)
        """
        B, C, h, w = LL.shape
        # 将 4 个子带 stack 回 (B,4C,h,w)
        y = torch.stack([LL, LH, HL, HH], dim=2).view(B, 4 * C, h, w)

        # conv_transpose2d 作为合成滤波器，stride=2
        weight = self.w_synthesis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        # conv_transpose 的权重形状：(in_channels, out_channels/groups, kH, kW)
        # 我们希望 groups=C，每组把 4 个子带合成为 1 个通道
        # 需要把 weight 视作 (4C, 1, 2, 2)，设置 groups=C 时会自动每4个输入映射到1个输出
        x_rec = F.conv_transpose2d(y, weight=weight, bias=None, stride=2, padding=0, groups=C)                                                                                                                                                                     # 微信公众号:AI缝合术
        return x_rec

    # ---------- 高频软阈值 ----------
    @staticmethod
    def soft_threshold(x, thr):
        # soft-shrinkage： sign(x) * relu(|x| - thr)
        return torch.sign(x) * F.relu(torch.abs(x) - thr)

    # ---------- 前向 ----------
    def forward(self, x):
        B, C, H, W = x.shape

        # 1) DWT
        LH, HL, HH, LL = self.dwt(x)

        # 2) 高频子带阈值化与融合
        # 归一化后的阈值（按通道），值域约束 0~1，再乘以该子带的平均幅度
        eps = 1e-6
        m_LH = LH.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HL = HL.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HH = HH.abs().mean(dim=(2, 3), keepdim=True) + eps

        t = torch.sigmoid(self.theta)  # (3,C,1,1)
        thr_LH = t[0].unsqueeze(0) * m_LH
        thr_HL = t[1].unsqueeze(0) * m_HL
        thr_HH = t[2].unsqueeze(0) * m_HH

        LH_hat = self.soft_threshold(LH, thr_LH)
        HL_hat = self.soft_threshold(HL, thr_HL)
        HH_hat = self.soft_threshold(HH, thr_HH)

        # 融合卷积（将 3C -> C）
        H_concat = torch.cat([LH_hat, HL_hat, HH_hat], dim=1)  # (B,3C,h,w)                                                                                                                                                                     # 微信公众号:AI缝合术
        H_fused = self.fuse(H_concat)  # (B,C,h,w)

        # 3) IDWT 重构
        X_re = self.idwt(LH_hat, HL_hat, H_fused, LL)  # (B,C,H',W')，H'/W'≈H/W                                                                                                                                                                     # 微信公众号:AI缝合术

        # 4) 注意力权重：GAP -> (可选FC) -> Softmax(沿通道)
        gap = F.adaptive_avg_pool2d(X_re, 1).view(B, C)  # (B,C)
        if self.use_fc:
            gap = self.fc(gap)  # (B,C)
        attn = F.softmax(gap, dim=1).view(B, C, 1, 1)  # (B,C,1,1)

        # 5) 加权原输入
        out = x * attn
        return out


class HWFEBlock(nn.Module):
    """
    Hybrid Stream Gating - X Block with HWFE
    混合流门控模块 - 使用 HWFE 替代 ConStream
    
    Args:
        c1: 输入通道数
        c2: 输出通道数
        alpha: 通道分割比例 (默认0.5)
        reduction: 降维比例 (默认16)
    """
    def __init__(self, c1, c2=None, alpha=0.5, reduction=8):
        super(HWFEBlock, self).__init__()
        
        if c2 is None:
            c2 = c1
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        
        # 计算分割通道数
        self.split_channels = int(c1 * alpha)
        self.gate_channels = c1 - self.split_channels

        self.reduced_channels = max(4, self.gate_channels // reduction)
        
        # 降维层
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(self.gate_channels, self.reduced_channels, 1),
            nn.BatchNorm2d(self.reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # 三个注意力流
        self.gct_stream = GctStream(self.reduced_channels)  # 全局上下文流
        self.rel_stream = RelStream(self.reduced_channels)  # 相对位置流
        self.hwfe_stream = WaveletAttention(self.reduced_channels)       # HWFE特征提取流（替代ConStream）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.reduced_channels * 2, self.reduced_channels, 1),
            nn.BatchNorm2d(self.reduced_channels),
            nn.SiLU()
        )
        
        # 扩展卷积
        self.expand_conv = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.gate_channels, 1),
            nn.BatchNorm2d(self.gate_channels)
        )
        
        # ⚠️ 最终卷积输入通道数调整
        concat_channels = self.split_channels + 2 * self.gate_channels
        self.final_conv = nn.Conv2d(concat_channels, c2, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # ===== 步骤1: 通道分割 =====
        identity_feat = x[:, :self.split_channels, :, :]  # αC
        gate_feat = x[:, self.split_channels:, :, :]      # (1-α)C
        
        # ===== 步骤2: 降维 =====
        reduced_feat = self.reduction_conv(gate_feat)
        
        # ===== 步骤3: 三个并行流 =====
        gct_out = self.gct_stream(reduced_feat)
        rel_out = self.rel_stream(reduced_feat)
        cons_out = self.hwfe_stream(reduced_feat)
        
        # 拼接和融合
        rel_cons = torch.cat([rel_out, cons_out], dim=1)
        rel_cons_fused = self.fusion_conv(rel_cons)
        
        # 调制
        interaction = gct_out * rel_cons_fused
        
        # 扩展
        interaction_expanded = self.expand_conv(interaction)
        
        # 残差
        enhanced_feat = gate_feat + interaction_expanded
        
        # ⚠️ 3分支拼接
        out = torch.cat([identity_feat, enhanced_feat, gate_feat], dim=1)
        
        # 最终输出
        out = self.final_conv(out)
        
        return out   
# --------------------------------------------------------------------------------------------------------  
class WaveletBottleneck(nn.Module):
    """
    小波注意力增强的Bottleneck
    标准Bottleneck + WaveletAttention
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        
        # 小波注意力
        self.wavelet = WaveletAttention(c2)
        
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = self.wavelet(out)  # 小波注意力增强
        return x + out if self.add else out

class C2f_WaveletBlock(nn.Module):
    """
    C2f风格的小波注意力模块
    
    优势:
    1. C2f梯度流 + 小波特征提取
    2. 多尺度频域特征复用
    3. 适合纹理丰富的场景
    
    Args:
        c1: 输入通道数
        c2: 输出通道数
        n: WaveletBottleneck重复次数
        shortcut: 是否使用残差连接
        g: 分组卷积数
        e: 通道扩展比例
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        
        # 初始卷积并分割
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        
        # WaveletBottleneck链
        self.m = nn.ModuleList(
            WaveletBottleneck(self.c, self.c, shortcut, g, k=((3, 3) if g == 1 else (1, 1)), e=1.0)
            for _ in range(n)
        )
        
        # 最终融合
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

    def forward(self, x):
        # 分割为两部分
        y = list(self.cv1(x).chunk(2, 1))
        
        # 逐层处理并保存所有中间特征
        y.extend(m(y[-1]) for m in self.m)
        
        # 拼接所有特征
        return self.cv2(torch.cat(y, 1))

class SKATBottleneck(nn.Module):
    """
    SKAT Transformer增强的Bottleneck
    包含SKA注意力 + KAN MLP
    """
    def __init__(self, c1, c2, num_heads=8, mlp_ratio=4.0, drop=0., drop_path=0., shortcut=True):
        super().__init__()
        
        # Layer Norm (需要转换为适合CNN的格式)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        
        # SKA注意力
        self.attn = SKA(c1, num_heads=num_heads, bias=False)
        
        # KAN MLP
        mlp_hidden_dim = int(c1 * mlp_ratio)
        self.mlp = KAN(
            in_features=c1,
            hidden_features=mlp_hidden_dim,
            out_features=c2,
            drop=drop,
            use_conv=False,
            act_init="gelu"
        )
        
        # DropPath
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.add = shortcut and c1 == c2

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 第一个block: Attention
        x_norm1 = x.flatten(2).permute(0, 2, 1)
        x_norm1 = self.norm1(x_norm1)
        x_norm1 = x_norm1.permute(0, 2, 1).view(B, C, H, W)
        
        attn_out = self.attn(x_norm1)
        x = x + self.drop_path1(attn_out)
        
        # 第二个block: MLP
        x_norm2 = x.flatten(2).permute(0, 2, 1)
        x_norm2 = self.norm2(x_norm2)
        x_norm2 = x_norm2.permute(0, 2, 1).view(B, C, H, W)
        
        mlp_out = self.mlp(x_norm2)
        
        if self.add:
            return x + self.drop_path2(mlp_out)
        else:
            return mlp_out


class C2f_SKATBlock(nn.Module):
    """
    C2f风格的SKAT Transformer模块
    
    优势:
    1. C2f梯度流 + SKAT注意力机制
    2. KAN MLP增强表达能力
    3. 多条梯度路径,训练稳定
    
    Args:
        c1: 输入通道数
        c2: 输出通道数
        n: SKATBottleneck重复次数
        num_heads: 注意力头数
        mlp_ratio: MLP扩展比例
        drop: Dropout比例
        drop_path: DropPath比例
        e: 通道扩展比例
    """
    def __init__(self, c1, c2, n=1, num_heads=8, mlp_ratio=4.0, drop=0., drop_path=0., e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        
        # 初始卷积并分割
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        
        # SKATBottleneck链
        self.m = nn.ModuleList(
            SKATBottleneck(
                self.c, self.c,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                shortcut=True
            )
            for _ in range(n)
        )
        
        # 最终融合
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

    def forward(self, x):
        # 分割为两部分
        y = list(self.cv1(x).chunk(2, 1))
        
        # 逐层处理并保存所有中间特征
        y.extend(m(y[-1]) for m in self.m)
        
        # 拼接所有特征
        return self.cv2(torch.cat(y, 1))  