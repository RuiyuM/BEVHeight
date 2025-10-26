from __future__ import annotations
from typing import List
import numpy as np
import torch
from active_learning.base import BaseActiveSelector


def _move_batch_to_device(sweep_imgs, mats, device):
    if torch.is_tensor(sweep_imgs):
        sweep_imgs = sweep_imgs.to(device, non_blocking=True)
    if isinstance(mats, dict):
        for k, v in mats.items():
            if torch.is_tensor(v):
                mats[k] = v.to(device, non_blocking=True)
    return sweep_imgs, mats


def _extract_U_map(backbone_out) -> torch.Tensor:
    """
    期望 backbone_out 为 (bev_feat, U_map) 或 (bev_feat, height, U_map) 或 (feat, H_img, U_map)
    统一取最后一个为 U_map，形状 [B, Hbev, Wbev]，且已在 [0,1]
    """
    if not (isinstance(backbone_out, tuple) and len(backbone_out) >= 2):
        raise RuntimeError("Backbone must return (..., U_map) when return_bev_bin_entropy=True.")
    U_map = backbone_out[-1]
    if not torch.is_tensor(U_map) or U_map.dim() != 3:
        raise RuntimeError(f"U_map must be [B,H,W] tensor, got {type(U_map)} with dim={getattr(U_map, 'dim', lambda: 'NA')()}")
    return U_map


def _score_from_U(U: torch.Tensor, top_ratio: float = 0.2) -> torch.Tensor:
    """
    U: [B, H, W] in [0,1]
    返回每张图的 S = 强度(Top-r均值) + 扩散(RMS 距离归一化)
    简化版：不做 M 的 valid 掩码
    """
    B, H, W = U.shape
    flat = U.view(B, -1)                            # [B, H*W]

    # --- Top-r 门槛（按值）---
    q = 1.0 - top_ratio
    # 兼容不支持 torch.quantile 的老版本：用 kthvalue 近似
    try:
        tau = torch.quantile(flat, q, dim=1, keepdim=True)  # [B,1]
    except AttributeError:
        k = max(1, int(q * (H * W)))
        tau = torch.topk(flat, k, dim=1, largest=False).values[:, -1:].clone()  # 第 q 分位的近似

    mask = (flat >= tau)                             # [B, H*W], bool
    # 强度：Top-r 的平均
    # 避免除 0：若没有元素（极端情况），强度记为 0
    sel_counts = mask.sum(dim=1).clamp_min(1)        # [B]
    intensity = (flat * mask.float()).sum(dim=1) / sel_counts  # [B] in [0,1]

    # 扩散：Top-r 坐标的 RMS 距离归一化
    # 先拿到坐标（归一化到 [0,1]）
    ys = torch.linspace(0, 1, H, device=U.device, dtype=U.dtype)
    xs = torch.linspace(0, 1, W, device=U.device, dtype=U.dtype)
    try:
        # 新版 PyTorch
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    except TypeError:
        # 兼容旧版（没有 indexing 参数，且默认 'ij'）
        yy, xx = torch.meshgrid(ys, xs)

    xx = xx.reshape(1, -1).expand(B, -1)  # [B, H*W]
    yy = yy.reshape(1, -1).expand(B, -1)

    # 只用 Top-r 的坐标计算质心与 RMS
    # 质心（不加权；你也可改成用 U 做权重）
    mask_f = mask.float()
    denom = sel_counts.float().unsqueeze(1)          # [B,1]
    cx = (xx * mask_f).sum(dim=1, keepdim=True) / denom   # [B,1]
    cy = (yy * mask_f).sum(dim=1, keepdim=True) / denom

    dx = (xx - cx).pow(2) * mask_f
    dy = (yy - cy).pow(2) * mask_f
    rms = torch.sqrt((dx + dy).sum(dim=1) / sel_counts.float())  # [B], ∈ [0, √2/2] 理论上界
    spread = rms / (np.sqrt(2.0) / 2.0 + 1e-12)            # 归一化到 [0,1]

    S = intensity + spread                                 # [B]，简单线性合成
    return S

class BinEntropyBEVMinSelector(BaseActiveSelector):
    """按 BEV bin-entropy 派生分数从小到大选 Top-K（对照组）"""

    @torch.no_grad()
    def select(
        self,
        model: torch.nn.Module,          # LightningModule
        dataloader,
        device: torch.device,
        unlabeled_indices: List[int],
        k: int,
    ) -> List[int]:
        model.eval()
        model.model.to(device)

        all_scores: List[float] = []

        for batch in dataloader:
            sweep_imgs, mats, _, img_metas, _, _ = batch
            sweep_imgs, mats = _move_batch_to_device(sweep_imgs, mats, device)

            # 从 backbone 取 BEV 熵图 U_map
            try:
                out = model.model.backbone(
                    sweep_imgs,
                    mats,
                    is_return_height=False,
                    return_bev_bin_entropy=True,
                )
            except TypeError:
                out = model.model(
                    sweep_imgs,
                    mats,
                    return_bev_bin_entropy=True,
                )

            U_map = _extract_U_map(out)              # [B, Hbev, Wbev], ∈[0,1]
            S = _score_from_U(U_map, top_ratio=0.2)  # [B]  强度+扩散（Top20%）
            all_scores.extend(S.detach().cpu().float().tolist())

        if len(all_scores) != len(unlabeled_indices):
            raise RuntimeError(
                f"score count {len(all_scores)} != pool size {len(unlabeled_indices)}; "
                "check dataloader indexing / subset."
            )

        # 反向：分数小优先
        order = np.argsort(all_scores)
        topk_rel = order[: min(k, len(order))].tolist()
        return [unlabeled_indices[i] for i in topk_rel]
class BinEntropyBEVSelector(BaseActiveSelector):
    """
    先把 bin-entropy 投到 BEV 得到 U_map，再用：
      - 强度 = Top-20% U 的均值
      - 扩散 = Top-20% U 像素的归一化 RMS 距离
    的和作为打分，按分数降序选 Top-K。
    """

    @torch.no_grad()
    def select(
        self,
        model: torch.nn.Module,          # LightningModule
        dataloader,
        device: torch.device,
        unlabeled_indices: List[int],
        k: int,
    ) -> List[int]:
        model.eval()
        model.model.to(device)

        all_scores: List[float] = []

        for batch in dataloader:
            sweep_imgs, mats, _, img_metas, _, _ = batch
            sweep_imgs, mats = _move_batch_to_device(sweep_imgs, mats, device)

            # 直接从 backbone 拿 U_map（不经过 head）
            try:
                out = model.model.backbone(
                    sweep_imgs,
                    mats,
                    is_return_height=False,
                    return_bev_bin_entropy=True,   # <<< 让 LSSFPN 返回 U_map
                )
            except TypeError:
                # 若你把开关透传到了 model(...) 也兼容
                out = model.model(
                    sweep_imgs,
                    mats,
                    return_bev_bin_entropy=True,
                )

            U_map = _extract_U_map(out)            # [B,Hbev,Wbev] in [0,1]
            S = _score_from_U(U_map, top_ratio=0.2)  # [B]
            all_scores.extend(S.detach().cpu().float().tolist())

        if len(all_scores) != len(unlabeled_indices):
            raise RuntimeError(
                f"score count {len(all_scores)} != pool size {len(unlabeled_indices)}; "
                "check dataloader indexing / subset."
            )

        order = np.argsort(all_scores)[::-1]   # 分数大优先
        topk_rel = order[: min(k, len(order))].tolist()
        return [unlabeled_indices[i] for i in topk_rel]