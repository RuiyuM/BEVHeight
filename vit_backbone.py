# 文件：projects/vit_backbone.py
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from mmcv.runner import BaseModule
from mmdet.models import BACKBONES              # ★ ← 用 mmdet 的注册器

@BACKBONES.register_module()
class ViTBackbone(BaseModule):
    """ViT backbone that outputs N feature maps for FPN/BEV."""
    def __init__(self,
                 arch='base',                # tiny / small / base / large
                 img_size=(256, 704),        # 训练 crop 大小
                 patch_size=16,
                 out_indices=(2, 5, 8, 11),  # 4 个层级
                 drop_path_rate=0.1,
                 init_cfg=None):
        super().__init__(init_cfg)
        embed_dim = dict(tiny=192, small=384, base=768, large=1024)[arch]
        depth = dict(tiny=12, small=12, base=12, large=24)[arch]
        num_heads = dict(tiny=3, small=6, base=12, large=16)[arch]

        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            qkv_bias=True,
            drop_path_rate=drop_path_rate,
            num_classes=0)
        self.out_indices = out_indices
        self.embed_dim = embed_dim

    def forward(self, x):
        B = x.size(0)
        x = self.vit.patch_embed(x)                # (B, N, C)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1) + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        outs = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.out_indices:
                feat = x[:, 1:, :].transpose(1, 2)  # (B, C, N)
                ph, pw = self.vit.patch_embed.patch_size  # ★ tuple
                H = self.vit.patch_embed.img_size[0] // ph
                W = self.vit.patch_embed.img_size[1] // pw
                outs.append(feat.reshape(B, self.embed_dim, H, W))
        return tuple(outs)
