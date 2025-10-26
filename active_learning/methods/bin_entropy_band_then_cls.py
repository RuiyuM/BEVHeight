# active_learning/methods/bin_entropy_band_then_cls.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
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


def _extract_H_img(backbone_out) -> torch.Tensor:
    """从 LSSFPN 的返回里取 H_img（图像级 bin-entropy）。"""
    if not isinstance(backbone_out, tuple) or len(backbone_out) < 2:
        raise RuntimeError("Backbone must return a tuple whose last element is H_img when return_bin_entropy=True.")
    H_img = backbone_out[-1]
    if not torch.is_tensor(H_img):
        raise RuntimeError("H_img must be a torch.Tensor.")
    return H_img.view(-1)  # [B]


# ---------- 映射构建：既支持 head.tasks，也支持 num_classes + class_names ----------
def _build_mapping_from_head_tasks(head) -> Optional[Dict[str, Tuple[int, int]]]:
    tasks = getattr(head, "tasks", None)
    if tasks is None:
        return None
    mapping: Dict[str, Tuple[int, int]] = {}
    for t, conf in enumerate(tasks):
        names = [n.lower() for n in conf.get("class_names", [])]
        for ci, name in enumerate(names):
            mapping[name] = (t, ci)
    return mapping


def _build_mapping_from_num_classes(head, class_names_all: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    用 head.num_classes（例如 [1,2,2,1,2,2]）和全局 CLASSES 顺序，推导每个类落在的 (task_idx, channel_idx)。
    """
    num = getattr(head, "num_classes", None)
    if num is None:
        raise RuntimeError("head.num_classes 未找到，无法推导通道映射。")
    # 累积偏移：task 起始全局 id
    starts = []
    s = 0
    for c in num:
        starts.append(s)
        s += int(c)
    name2gid = {n.lower(): i for i, n in enumerate(class_names_all)}
    mapping: Dict[str, Tuple[int, int]] = {}

    def _pos(global_id: int) -> Tuple[int, int]:
        # 找到 global_id 属于哪个 task 的区间
        for t, s0 in enumerate(starts):
            c = num[t]
            if s0 <= global_id < s0 + c:
                return t, global_id - s0
        raise RuntimeError(f"global id {global_id} 不在任何 task 区间内。")

    for name, gid in name2gid.items():
        mapping[name] = _pos(gid)
    return mapping


def extract_logits_car_cyc_ped_fixed(preds):
    """
    preds: list[dict]，len=6，每个 dict 含 'heatmap': [B, C_t, H, W]
    返回: [B, 3, H, W]，顺序 [Car, Cyclist, Pedestrian]
    """
    # Car: task 0, ch 0
    logit_car = preds[0]['heatmap'][:, 0:1]  # [B,1,H,W]

    # Cyclist = logsumexp(motorcycle, bicycle)
    # task 4: ['motorcycle','bicycle'] → ch 0 和 ch 1
    hm_cyc2 = preds[4]['heatmap'][:, 0:2]    # [B,2,H,W]
    logit_cyc = torch.logsumexp(hm_cyc2, dim=1, keepdim=True)  # [B,1,H,W]

    # Pedestrian: task 5, ch 0  （['pedestrian','traffic_cone']）
    logit_ped = preds[5]['heatmap'][:, 0:1]  # [B,1,H,W]

    return torch.cat([logit_car, logit_cyc, logit_ped], dim=1)  # [B,3,H,W]


class BinEntropyBandThenLogitClsSelector(BaseActiveSelector):
    """
    两阶段选择：
      1) 计算每图 bin-entropy（来自 LSSFPN；高度/深度分布熵的平均），分桶（top/mid/bottom，默认30%）。
      2) 只在该桶内，按 Car/Cyclist/Ped 三类 logits 的图像级熵从高到低取 Top-K。
    """

    def __init__(self, band: str = "top", band_ratio: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        band = band.lower()
        assert band in ("top", "mid", "bottom")
        assert 0.0 < band_ratio < 0.5, "band_ratio 建议在 (0, 0.5)，例如 0.3"
        self.band = band
        self.band_ratio = float(band_ratio)

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

        # -------- Pass 1: 每图 bin-entropy --------
        H_bin_all: List[float] = []
        for batch in dataloader:
            sweep_imgs, mats, _, img_metas, _, _ = batch
            sweep_imgs, mats = _move_batch_to_device(sweep_imgs, mats, device)
            try:
                out = model.model.backbone(
                    sweep_imgs, mats, is_return_height=False, return_bin_entropy=True
                )
            except TypeError:
                out = model.model(sweep_imgs, mats, return_bin_entropy=True)
            H_img = _extract_H_img(out)  # [B]
            H_bin_all.extend(H_img.detach().cpu().float().tolist())

        if len(H_bin_all) != len(unlabeled_indices):
            raise RuntimeError(f"[bin-entropy] score count {len(H_bin_all)} != pool size {len(unlabeled_indices)}")

        H_bin_all = np.asarray(H_bin_all, dtype=np.float32)

        # 分位数分桶
        r = self.band_ratio
        q_lo = np.quantile(H_bin_all, r)
        q_hi = np.quantile(H_bin_all, 1.0 - r)
        q_mlo = np.quantile(H_bin_all, 0.5 - r / 2.0)
        q_mhi = np.quantile(H_bin_all, 0.5 + r / 2.0)

        if self.band == "top":
            mask = (H_bin_all >= q_hi)
        elif self.band == "bottom":
            mask = (H_bin_all <= q_lo)
        else:
            mask = (H_bin_all >= q_mlo) & (H_bin_all <= q_mhi)

        cand_idx_rel = np.nonzero(mask)[0].tolist()
        if not cand_idx_rel:
            cand_idx_rel = list(range(len(unlabeled_indices)))  # 极端退化

        # -------- Pass 2: 桶内按分类熵排序 --------
        H_cls_all: List[float] = []
        for batch in dataloader:
            sweep_imgs, mats, _, img_metas, _, _ = batch
            sweep_imgs, mats = _move_batch_to_device(sweep_imgs, mats, device)
            preds = model.model(sweep_imgs, mats)  # list[dict] x num_tasks
            logits_3 = extract_logits_car_cyc_ped_fixed(preds, model)  # [B,3,H,W]
            p = torch.softmax(logits_3, dim=1).clamp_min(1e-12)
            H_map = -(p * p.log()).sum(dim=1)  # [B,H,W]
            H_img = H_map.mean(dim=(1, 2))    # [B]
            H_cls_all.extend(H_img.detach().cpu().float().tolist())

        if len(H_cls_all) != len(unlabeled_indices):
            raise RuntimeError(f"[cls-entropy] score count {len(H_cls_all)} != pool size {len(unlabeled_indices)}")

        H_cls_all = np.asarray(H_cls_all, dtype=np.float32)

        cand_scores = H_cls_all[cand_idx_rel]
        order_rel_in_cand = np.argsort(-cand_scores)  # 降序：熵大更不确定
        pick_rel = order_rel_in_cand[:min(k, len(order_rel_in_cand))].tolist()
        chosen_rel = [cand_idx_rel[i] for i in pick_rel]

        return [unlabeled_indices[i] for i in chosen_rel]


# 可选：注册到方法表
try:
    from active_learning.methods import _METHOD_REGISTRY
    _METHOD_REGISTRY["bin_entropy_band_then_cls_top"] = lambda **kw: BinEntropyBandThenLogitClsSelector(band="top", **kw)
    _METHOD_REGISTRY["bin_entropy_band_then_cls_mid"] = lambda **kw: BinEntropyBandThenLogitClsSelector(band="mid", **kw)
    _METHOD_REGISTRY["bin_entropy_band_then_cls_bottom"] = lambda **kw: BinEntropyBandThenLogitClsSelector(band="bottom", **kw)
except Exception:
    pass
