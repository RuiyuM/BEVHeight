# active_learning/methods/gt_diversity_greedy.py
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from active_learning.base import BaseActiveSelector
from dataset.nusc_mv_det_dataset import collate_fn

# ---- 超参 ----
_EPS = 1e-8

# 你的 10 类到 3 元类的映射
_ATOM2META = {
    0: "car",          # car
    8: "ped",          # pedestrian
    6: "cyc", 7: "cyc" # motorcycle, bicycle
}
_META_ORDER = ["car", "ped", "cyc"]  # 固定顺序


# =============== 工具函数 ===============

def _move_to_dev(imgs, mats, dev):
    if torch.is_tensor(imgs): imgs = imgs.to(dev, non_blocking=True)
    if isinstance(mats, dict):
        for k, v in mats.items():
            if torch.is_tensor(v): mats[k] = v.to(dev, non_blocking=True)
    return imgs, mats

def _entropy(p: torch.Tensor) -> torch.Tensor:
    # p: [..., D], 已是概率分布
    p = p.clamp_min(_EPS)
    H = -(p * p.log()).sum(dim=-1)
    D = p.shape[-1]
    return H / np.log(D)  # 归一化到 [0,1]

def _jsd(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    # P, Q: [D], 概率分布
    M = 0.5 * (P + Q)
    kl = lambda A, B: (A.clamp_min(_EPS) * (A.clamp_min(_EPS) / B.clamp_min(_EPS)).log()).sum(-1)
    J = 0.5 * (kl(P, M) + kl(Q, M))
    D = P.shape[-1]
    return J / np.log(D)  # 归一化到 [0,1]

def _build_roi_mask_from_gt(
    bev_W: int, bev_H: int, voxel_coord_xy: torch.Tensor, voxel_size_xy: torch.Tensor,
    gt_boxes: torch.Tensor, gt_labels: torch.Tensor, want_meta: str,
) -> torch.Tensor:
    """
    返回 BEV 布尔掩码 [H, W]，用 GT 椭圆 ROI（中心(cx,cy), 半径 w/2, l/2, 角度 yaw）覆盖。
    - voxel_coord_xy: [2] = (x0_center, y0_center)
    - voxel_size_xy:  [2] = (vx, vy)
    - gt_boxes: [N, >=7] with [x, y, z, w, l, h, yaw, ...]
    - gt_labels: [N]
    """
    if gt_boxes.numel() == 0:
        return torch.zeros(bev_H, bev_W, dtype=torch.bool, device=gt_boxes.device)

    device = gt_boxes.device
    x0, y0 = float(voxel_coord_xy[0].item()), float(voxel_coord_xy[1].item())
    vx, vy = float(voxel_size_xy[0].item()), float(voxel_size_xy[1].item())

    # 选出属于当前元类的框
    keep = []
    for i in range(gt_labels.numel()):
        lbl = int(gt_labels[i].item())
        if lbl in _ATOM2META and _ATOM2META[lbl] == want_meta:
            keep.append(i)
    if not keep:
        return torch.zeros(bev_H, bev_W, dtype=torch.bool, device=device)

    boxes = gt_boxes[keep]  # [M, >=7]
    mask = torch.zeros(bev_H, bev_W, dtype=torch.bool, device=device)

    for b in boxes:
        cx, cy, _, w, l, _, yaw = [float(v) for v in b[:7]]
        # 最大半径 -> 限定索引窗口
        r = 0.5 * (w**2 + l**2) ** 0.5
        ix_c = int(round((cx - x0) / vx))
        iy_c = int(round((cy - y0) / vy))
        rx = int(np.ceil(r / vx))
        ry = int(np.ceil(r / vy))
        xL, xR = max(0, ix_c - rx), min(bev_W - 1, ix_c + rx)
        yT, yB = max(0, iy_c - ry), min(bev_H - 1, iy_c + ry)
        if xL > xR or yT > yB:
            continue

        # 网格中心的真实坐标
        xs = x0 + torch.arange(xL, xR + 1, device=device, dtype=torch.float32) * vx
        ys = y0 + torch.arange(yT, yB + 1, device=device, dtype=torch.float32) * vy
        YY, XX = torch.meshgrid(ys, xs)  # [h, w]
        # 旋转到框坐标系
        c, s = np.cos(yaw), np.sin(yaw)
        dx = (XX - cx) * c + (YY - cy) * s
        dy = -(XX - cx) * s + (YY - cy) * c
        # 椭圆内测试（更稳健，不依赖 w/l 轴顺序约定）
        hx = max(w * 0.5, 1e-3)
        hy = max(l * 0.5, 1e-3)
        inside = (dx / hx) ** 2 + (dy / hy) ** 2 <= 1.0
        mask[yT:yB + 1, xL:xR + 1] |= inside

    return mask  # [H, W] bool


def _aggregate_Q_for_one_sample(
    height_pd: torch.Tensor,        # [C, D, fH, fW]
    geom_idx_xy: torch.Tensor,      # [C, D, fH, fW, 2], int
    roi_mask: torch.Tensor,         # [H_bev, W_bev] bool
) -> Tuple[torch.Tensor, float]:
    """
    返回 (Q, n)：
      - Q: [D] 该 ROI 的 depth/height 分布，所有相机/像素聚合的加权平均
      - n: 标量，权重总和（像素的“命中 bin”数）
    聚合策略：对每个像素 (u,v,cam)，以“其投到 ROI 内的 bin 数”为权重，平均它的整条 p(d)。
    """
    device = height_pd.device
    C, D, fH, fW = height_pd.shape
    # 有效的 BEV 索引范围
    W_bev = int(roi_mask.shape[1])
    H_bev = int(roi_mask.shape[0])

    x_idx = geom_idx_xy[..., 0].clamp(-1, W_bev)  # [C, D, fH, fW]
    y_idx = geom_idx_xy[..., 1].clamp(-1, H_bev)
    valid = (x_idx >= 0) & (x_idx < W_bev) & (y_idx >= 0) & (y_idx < H_bev)
    if not valid.any():
        return torch.zeros(D, device=device), 0.0

    # 将 ROI mask 展平后做索引
    roi_flat = roi_mask.view(-1)  # [H*W]
    flat = (y_idx * W_bev + x_idx).view(C, D, fH, fW)  # [C,D,fH,fW]
    in_roi = torch.zeros_like(valid, dtype=torch.bool)
    # 只在 valid 元素处做查表
    in_roi[valid] = roi_flat[flat[valid].long()]

    # 每像素的“命中 bin 数”作为权重
    w_pix = in_roi.sum(dim=1).float()  # [C, fH, fW]
    w_sum = w_pix.sum().item()
    if w_sum <= 0:
        return torch.zeros(D, device=device), 0.0

    # 聚合整条分布：Σ (w_pix * p(d)) / Σ w_pix
    Q = (height_pd * w_pix.unsqueeze(1)).sum(dim=(0, 2, 3))  # [D]
    Q = (Q / (Q.sum() + _EPS)).contiguous()
    return Q, float(w_sum)


def _compute_class_proto_from_loader(model, dl, device) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    用“已标注集”估计每个元类的原型分布 P_c，返回：
      P: dict meta-> [D] 张量
      N: dict meta-> 权重总和（float）
    """
    model.eval(); model.model.to(device)
    P = {m: None for m in _META_ORDER}
    N = {m: 0.0 for m in _META_ORDER}

    # 取 BEV 网格参数
    vxvy = model.model.backbone.voxel_size[:2]          # [2]
    vcoord = model.model.backbone.voxel_coord[:2]       # [2]
    bev_W = int(model.model.backbone.voxel_num[0].item())
    bev_H = int(model.model.backbone.voxel_num[1].item())

    for batch in dl:
        sweep_imgs, mats, _, img_metas, gt_boxes_l, gt_labels_l = batch
        B = sweep_imgs.shape[0]
        sweep_imgs, mats = _move_to_dev(sweep_imgs, mats, device)

        out = model.model.backbone(
            sweep_imgs, mats,
            is_return_height=True,
            return_geom_index=True,
        )  # (bev_feat, height, geom_idx_xy)
        _, height_all, geom_all = out
        # 还原到 [B,C,D,fH,fW]
        num_cams = mats['intrin_mats'].shape[2]
        height_all = height_all.view(B, num_cams, *height_all.shape[1:])
        geom_all = geom_all  # [B,C,D,fH,fW,2]

        for b in range(B):
            # GT
            gt_b = gt_boxes_l[b].detach().cpu()
            gt_l = gt_labels_l[b].detach().cpu()

            # 每个元类：ROI -> Q, n
            for meta in _META_ORDER:
                roi = _build_roi_mask_from_gt(
                    bev_W, bev_H, vcoord, vxvy,
                    gt_b, gt_l, meta
                ).to(device)

                Q, n = _aggregate_Q_for_one_sample(
                    height_all[b], geom_all[b], roi
                )
                if n > 0:
                    if P[meta] is None:
                        P[meta] = Q
                    else:
                        P[meta] = (P[meta] * N[meta] + Q * n) / (N[meta] + n + _EPS)
                    N[meta] += n

    # 缺失类用均匀分布兜底
    for meta in _META_ORDER:
        if P[meta] is None:
            D = int(model.model.backbone.height_channels)
            P[meta] = torch.full((D,), 1.0 / D, device=device)
    return P, N


# =============== 主选择器（贪心） ===============

class _GTDiverseGreedyBase(BaseActiveSelector):
    """
    用 GT 框构造 BEV ROI，反查到 (cam,u,v,bin) 的 p(d)：
      - 先从“已标注集”估计每个元类原型分布 P_c
      - 对未标注池中每张图，计算每类的 Q_{i,c} 与权重 n_{i,c}
      - 贪心：每次选 1 张（按 S_i 打分），并在线更新 P_c <- (P_c*N + Q*n)/(N+n)
    子类通过 self.maximize 控制是“多样性最大优先”还是“最小优先”
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.maximize = True  # 子类覆盖

    @torch.no_grad()
    def select(self, model, dataloader, device, unlabeled_indices: List[int], k: int) -> List[int]:
        model.eval(); model.model.to(device)
        # ---------- 1) 从已标注集估计 P_c ----------
        ds = model._train_dataset if model._train_dataset is not None else model._build_train_dataset()
        labeled = list(getattr(model, "_labeled_indices", []) or [])
        if labeled:
            dl_lab = DataLoader(Subset(ds, labeled),
                                batch_size=model.batch_size_per_device,
                                num_workers=2,
                                shuffle=False,
                                collate_fn=lambda *a, **kw: collate_fn(*a, is_return_depth=False))
            P, N = _compute_class_proto_from_loader(model, dl_lab, device)
        else:
            # 无已标注：用均匀分布初始化
            D = int(model.model.backbone.height_channels)
            P = {m: torch.full((D,), 1.0 / D, device=device) for m in _META_ORDER}
            N = {m: 0.0 for m in _META_ORDER}

        # BEV 网格参数
        vxvy = model.model.backbone.voxel_size[:2]
        vcoord = model.model.backbone.voxel_coord[:2]
        bev_W = int(model.model.backbone.voxel_num[0].item())
        bev_H = int(model.model.backbone.voxel_num[1].item())

        # ---------- 2) 预计算未标注池每张图的 Q_{i,c}, n_{i,c} ----------
        Q_list: List[Dict[str, torch.Tensor]] = []
        n_list: List[Dict[str, float]] = []

        for batch in dataloader:
            sweep_imgs, mats, _, img_metas, gt_boxes_l, gt_labels_l = batch
            B = sweep_imgs.shape[0]
            sweep_imgs, mats = _move_to_dev(sweep_imgs, mats, device)

            out = model.model.backbone(
                sweep_imgs, mats,
                is_return_height=True,
                return_geom_index=True,
            )
            _, height_all, geom_all = out
            num_cams = mats['intrin_mats'].shape[2]
            height_all = height_all.view(B, num_cams, *height_all.shape[1:])  # [B,C,D,fH,fW]

            for b in range(B):
                qi: Dict[str, torch.Tensor] = {}
                ni: Dict[str, float] = {}
                gt_b = gt_boxes_l[b].detach().cpu()
                gt_l = gt_labels_l[b].detach().cpu()
                for meta in _META_ORDER:
                    roi = _build_roi_mask_from_gt(
                        bev_W, bev_H, vcoord, vxvy,
                        gt_b, gt_l, meta
                    ).to(device)

                    Q, n = _aggregate_Q_for_one_sample(
                        height_all[b], geom_all[b], roi
                    )
                    if n > 0:
                        qi[meta] = Q
                        ni[meta] = n
                Q_list.append(qi)
                n_list.append(ni)

        if len(Q_list) != len(unlabeled_indices):
            raise RuntimeError(f"pool size mismatch: feats={len(Q_list)} vs indices={len(unlabeled_indices)}")

        # ---------- 3) 贪心选择 ----------
        chosen_rel = []
        alive = set(range(len(unlabeled_indices)))

        def score_one(i: int) -> float:
            qi, ni = Q_list[i], n_list[i]
            if not qi:
                return -1e9 if self.maximize else 1e9  # 无三类目标，按极端分数处理
            s_vals = []
            for meta in qi.keys():
                s_vals.append(float(_jsd(P[meta], qi[meta]).item()))  # JSD(Q||P)
            return float(np.mean(s_vals))

        for _ in range(min(k, len(unlabeled_indices))):
            # 评估所有未选
            cand = list(alive)
            scores = np.array([score_one(i) for i in cand], dtype=np.float32)
            if self.maximize:
                j = int(cand[int(scores.argmax())])
            else:
                j = int(cand[int(scores.argmin())])

            chosen_rel.append(j)
            alive.remove(j)

            # 在线更新 P_c
            qi, ni = Q_list[j], n_list[j]
            for meta in qi.keys():
                n_old = N[meta]
                P[meta] = (P[meta] * n_old + qi[meta] * ni[meta]) / (n_old + ni[meta] + _EPS)
                N[meta] = n_old + ni[meta]

        return [unlabeled_indices[i] for i in chosen_rel]


class GTDiverseGreedyMaxSelector(_GTDiverseGreedyBase):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.maximize = True


class GTDiverseGreedyMinSelector(_GTDiverseGreedyBase):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.maximize = False


# 自动注册（可选）
try:
    from active_learning.methods import _METHOD_REGISTRY
    _METHOD_REGISTRY["gt_diverse_greedy_max"] = GTDiverseGreedyMaxSelector
    _METHOD_REGISTRY["gt_diverse_greedy_min"] = GTDiverseGreedyMinSelector
except Exception:
    pass
