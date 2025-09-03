# vggt_backbone.py  —— 简洁版（4D 输入, ImageNet 归一化假设）
import os, sys, math, importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==== 导入 VGGT（按需设置你的本地源码路径）====
FALLBACK_VGGT_SRC = "/people/cs/r/rxm210041/Desktop/test_3d_active/vggt"
try:
    import vggt  # noqa: F401
except Exception:
    if os.path.isdir(FALLBACK_VGGT_SRC):
        sys.path.insert(0, FALLBACK_VGGT_SRC)
    import vggt  # noqa: F401

from vggt.models.vggt import VGGT

# 让 vggt 源码里可能缓存的 checkpoint 符号指向入口脚本已打补丁的版本（若存在）
try:
    import torch.utils.checkpoint as _tucp
    vt = importlib.import_module("vggt.layers.vision_transformer")
    vt.checkpoint = getattr(_tucp, "checkpoint", vt.checkpoint)
except Exception:
    pass

# ==== mmdet registry 兼容导入 ====
try:
    from mmcv.runner import BaseModule          # mmcv 1.x
except Exception:
    from mmengine.model import BaseModule       # mmengine

try:
    from mmdet.registry import MODELS as MMDET_MODELS  # 新式
except Exception:
    MMDET_MODELS = None
try:
    from mmdet.models import BACKBONES as MMDET_BACKBONES  # 旧式
except Exception:
    MMDET_BACKBONES = None

PATCH = 14  # VGGT patch size

def pad_to_multiple(x, mult=PATCH, mode="replicate"):
    B, C, H, W = x.shape
    ph = (mult - H % mult) % mult
    pw = (mult - W % mult) % mult
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x, ph, pw

def tokens_to_map(tokens, ps_idx, h, w):
    """
    tokens: [B, T, C]
    ps_idx: 可能是 [B,T] (flatten idx) 或 [B,T,2] (y,x)；也可能为 None（则尝试 reshape）
    return: [B, C, h, w]
    """
    B, T, C = tokens.shape
    dev, dt = tokens.device, tokens.dtype

    if T == h * w:
        return tokens.transpose(1, 2).reshape(B, C, h, w)

    if ps_idx is not None:
        if ps_idx.dim() == 3 and ps_idx.size(-1) == 2:
            y = ps_idx[..., 0].long()
            x = ps_idx[..., 1].long()
            idx = (y * w + x).long()  # [B,T]
        else:
            idx = ps_idx.long()

        grid = torch.zeros(B, C, h * w, device=dev, dtype=dt)
        grid.scatter_add_(2, idx.unsqueeze(1).expand(B, C, T), tokens.transpose(1, 2))

        ones = torch.ones(B, 1, T, device=dev, dtype=dt)
        cnt = torch.zeros(B, 1, h * w, device=dev, dtype=dt)
        cnt.scatter_add_(2, idx.unsqueeze(1), ones)
        grid = grid / cnt.clamp_min(1.0)
        return grid.view(B, C, h, w)

    # fallback：尽量 reshape（截断多余 token 或零填充）
    fmap = torch.zeros(B, C, h * w, device=dev, dtype=dt)
    take = min(T, h * w)
    fmap[:, :, :take] = tokens[:, :take, :].transpose(1, 2)
    return fmap.view(B, C, h, w)


class _VGGTCore(BaseModule):
    def __init__(self,
                 pretrained_id='facebook/VGGT-1B',
                 out_channels=256,
                 freeze_vggt=True,
                 init_cfg=None,
                 # ↓↓↓ 这些是“可选/兼容键”，即使传了也不会报错
                 img_size=None,
                 pad_multiple=14,
                 use_amp=True,
                 mode='pyramid',
                 out_indices=None,
                 img_mean=None,
                 img_std=None,
                 to_rgb=True,
                 **kwargs):
        super().__init__(init_cfg)
        self.vggt = VGGT.from_pretrained(pretrained_id)
        if freeze_vggt:
            for p in self.vggt.parameters():
                p.requires_grad = False
        self.out_channels = out_channels
        self._proj = nn.ModuleList([nn.Identity()] * 4)
        self._proj_ready = False
        # 把可用的值存起来（即便当前简洁版没用到其它键，也不报错）
        self.pad_multiple = pad_multiple
        self.use_amp = use_amp

    def init_weights(self):
        return

    @torch.no_grad()
    def _ensure_proj(self, fmap_like: torch.Tensor):
        dev = fmap_like.device
        dt = fmap_like.dtype

        if self._proj_ready:
            # 如果早就建过，但设备/精度不一致，还是要搬一次
            for m in self._proj:
                m.to(device=dev, dtype=dt)
            return

        Ctok = fmap_like.shape[1]
        self._proj = nn.ModuleList([nn.Conv2d(Ctok, self.out_channels, 1) for _ in range(4)])
        # ★ 新建后立刻搬到与输入相同的 device/dtype
        for m in self._proj:
            m.to(device=dev, dtype=dt)
        self._proj_ready = True

    def forward(self, x):
        """
        x: [B,3,H,W]  —— 已是 ImageNet 规范化的张量
        目标：输出4个层级，每层 [B, 192, 54, 96]（以 H=864, W=1536 举例）
        """
        B, C, H, W = x.shape

        # 1) pad 到 14 的倍数（VGGT patch）
        x_pad, ph, pw = pad_to_multiple(x, self.pad_multiple if hasattr(self, "pad_multiple") else 14)
        Hpad, Wpad = x_pad.shape[-2:]
        h = math.ceil(Hpad / 14)
        w = math.ceil(Wpad / 14)

        # 2) aggregator 要 5D；你的 tokens_list 是 24 层
        x5 = x_pad if x_pad.dim() == 5 else x_pad.unsqueeze(1)
        with torch.no_grad():
            tokens_list, ps_idx_raw = self.vggt.aggregator(x5)

        # 工具：仅在是张量时才“去掉序列维”，否则原样返回
        def squeeze_seq_tensor(t):
            if torch.is_tensor(t):
                return t[:, 0] if t.dim() == 4 else t  # [B,T,C] / [B,T] / [B,T,2]
            return t

        # 统一“拿第 i 层的 ps_idx”：若不是张量（如 int/None），返回 None 走 fallback
        def pick_ps_idx(ps_idx_any, i):
            item = ps_idx_any
            if isinstance(ps_idx_any, (list, tuple)) and len(ps_idx_any) > 0:
                item = ps_idx_any[i if i < len(ps_idx_any) else -1]
            if isinstance(item, (list, tuple)):
                # 如果还是容器，尽力找第一个张量
                for sub in item:
                    if torch.is_tensor(sub):
                        item = sub
                        break
            return squeeze_seq_tensor(item) if torch.is_tensor(item) else None

        L = len(tokens_list)
        # 3) 选择4个层（对齐 ViT out_indices 思路）
        if getattr(self, "out_indices", None) is None:
            idxs = [max(1, int(L * 0.25) - 1), int(L * 0.5) - 1, int(L * 0.75) - 1, L - 1]
        else:
            idxs = [i if i >= 0 else (L + i) for i in self.out_indices]
            idxs = [min(max(i, 0), L - 1) for i in idxs]

        fmap_list = []
        for i in idxs:
            toks = squeeze_seq_tensor(tokens_list[i])  # -> [B,T,Ctok]
            ps = pick_ps_idx(ps_idx_raw, i)  # -> [B,T]/[B,T,2] 或 None

            # 4) tokens -> fmap（无 ps_idx 时裁/填到 h*w）
            B2, T, Ctok = toks.shape
            if (ps is None) or (not torch.is_tensor(ps)):
                # 无索引：裁/填到 h*w，然后 reshape
                Thw = h * w
                if T >= Thw:
                    toks_use = toks[:, :Thw, :]
                else:
                    pad = Thw - T
                    toks_use = torch.cat([toks, toks.new_zeros(B2, pad, Ctok)], dim=1)
                fmap = toks_use.transpose(1, 2).reshape(B2, Ctok, h, w)
            else:
                # 有索引：按 idx/scatter 复原（你已有 tokens_to_map）
                fmap = tokens_to_map(toks, ps, h, w)

            fmap_list.append(fmap)

        # 第一次运行后初始化 1×1 conv，把 Ctok 映到 192
        self._ensure_proj(fmap_list[0])  # 传入张量以获知 Ctok

        # 5) 统一到 (H//16, W//16) = (54, 96)，再做 1×1 降通道
        target = (H // 16, W // 16)  # 864/16=54, 1536/16=96
        outs = []
        for i in range(4):
            y = F.interpolate(fmap_list[i], size=target, mode='bilinear', align_corners=False)
            y = self._proj[i](y)  # -> [B, 192, 54, 96]（若 out_channels=192）
            outs.append(y)

        return outs


# ====== 注册到 mmdet（新/旧二选一；都没有时也能 import 成功）======
if MMDET_MODELS is not None:
    @MMDET_MODELS.register_module()
    class VGGTBackbone(_VGGTCore):
        pass
elif MMDET_BACKBONES is not None:
    @MMDET_BACKBONES.register_module()
    class VGGTBackbone(_VGGTCore):
        pass
else:
    class VGGTBackbone(_VGGTCore):
        pass
