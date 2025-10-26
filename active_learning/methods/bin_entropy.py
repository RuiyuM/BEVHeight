# active_learning/methods/bin_entropy.py
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch
import os, csv, time

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
    """
    backbone_out 期望为：
      - (bev_feat, H_img)
      - (bev_feat, height, H_img)
      - (preds, H_img)
    统一取最后一个为 H_img: [B]
    """
    if not isinstance(backbone_out, tuple) or len(backbone_out) < 2:
        raise RuntimeError(
            "Backbone must return a tuple whose last element is H_img when return_bin_entropy=True."
        )
    H_img = backbone_out[-1]
    if not torch.is_tensor(H_img):
        raise RuntimeError("H_img must be a torch.Tensor.")
    if H_img.dim() == 0:
        H_img = H_img.view(1)
    return H_img


def _count_cpc_from_batch(gt_labels) -> Tuple[List[int], List[int], List[int]]:
    """
    统计当前 batch 每张图的三类数量（只考虑 current_classes=["Car","Pedestrian","Cyclist"]）：
      Car:        label in {0 (car), 3 (bus)}         # 把 bus 合并到 Car
      Pedestrian: label == 8
      Cyclist:    label in {7 (bicycle), 6 (motorcycle)}  # 目前常见 7；若将来有 6 也计入
    其他类别忽略。
    支持 list[Tensor] / list[np.ndarray] / Tensor（张量时无法拆分样本，返回空列表让上游回退为 0）。
    """
    CAR_IDS = {0, 3}
    PED_IDS = {8}
    CYC_IDS = {6, 7}

    cars, peds, cycs = [], [], []
    if gt_labels is None:
        return cars, peds, cycs

    if isinstance(gt_labels, (list, tuple)):
        for lab in gt_labels:
            if lab is None:
                cars.append(0); peds.append(0); cycs.append(0); continue
            if torch.is_tensor(lab):
                lab = lab.detach().cpu().long()
            else:
                lab = torch.as_tensor(lab, dtype=torch.long)
            if lab.numel() == 0:
                cars.append(0); peds.append(0); cycs.append(0); continue

            cars.append(int(sum((lab == i).sum().item() for i in CAR_IDS)))
            peds.append(int(sum((lab == i).sum().item() for i in PED_IDS)))
            cycs.append(int(sum((lab == i).sum().item() for i in CYC_IDS)))
        return cars, peds, cycs

    # 若是单个 Tensor（无 per-image 切分信息），返回空表让上层回退为 0
    return cars, peds, cycs


def _ensure_log_dir(model, log_dir: Optional[str]) -> str:
    base = log_dir or os.path.join(getattr(model, "default_root_dir", "."), "al_logs")
    os.makedirs(base, exist_ok=True)
    return base


def _append_rows_csv(csv_path: str, rows: List[dict], write_header_if_new: bool = True):
    is_new = (not os.path.exists(csv_path))
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iter_ts", "global_idx", "depth_entropy",
                "gt_car", "gt_ped", "gt_cyc"
            ],
        )
        if write_header_if_new and is_new:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


class _BinEntropyBase(BaseActiveSelector):
    """基类：计算 H_img，并把 (idx, H, GT计数) 记录到 CSV。"""

    def __init__(self, log_dir: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._log_dir = log_dir  # 可外部传入；否则落到 model.default_root_dir/al_logs

    def _gather_and_log(
        self,
        model,
        dataloader,
        device,
        unlabeled_indices: List[int],
    ) -> List[float]:
        """遍历 pool，计算 H_img，同时记录统计 CSV。返回 per-sample 的 H 列表，与 unlabeled_indices 对齐。"""
        model.eval()
        model.model.to(device)

        all_H: List[float] = []
        rows: List[dict] = []
        ts = int(time.time())
        csv_dir = _ensure_log_dir(model, self._log_dir)
        csv_path = os.path.join(csv_dir, "depth_entropy_stats.csv")

        pool_ptr = 0  # 与 unlabeled_indices 对齐

        for batch in dataloader:
            # 典型 batch: (sweep_imgs, mats, _, img_metas, gt_boxes, gt_labels)
            # 若无 GT 则可能是 (sweep_imgs, mats, _, img_metas, _, _)
            if len(batch) >= 6:
                sweep_imgs, mats, _, img_metas, gt_boxes, gt_labels = batch
            else:
                sweep_imgs, mats = batch[0], batch[1]
                gt_boxes, gt_labels = None, None

            if torch.is_tensor(sweep_imgs):
                B = sweep_imgs.shape[0]
            else:
                # 回退（几乎不会触发）
                B = len(unlabeled_indices) - pool_ptr

            sweep_imgs, mats = _move_batch_to_device(sweep_imgs, mats, device)

            # 直接从 backbone 拿图像级熵（不经过 head）
            try:
                backbone_out = model.model.backbone(
                    sweep_imgs, mats,
                    is_return_height=False,
                    return_bin_entropy=True,
                )
            except TypeError:
                backbone_out = model.model(
                    sweep_imgs, mats,
                    return_bin_entropy=True,
                )

            H_img = _extract_H_img(backbone_out).detach().cpu().float()  # [B]
            all_H.extend(H_img.tolist())

            # 统计 GT 三类数量（允许无 GT 的设定，计 0）
            cars, peds, cycs = _count_cpc_from_batch(gt_labels)
            if (len(cars) == 0) and (B > 0):
                cars = [0] * B; peds = [0] * B; cycs = [0] * B

            # 写入行（global_idx 按 dataloader 顺序映射 unlabeled_indices）
            for j in range(B):
                if pool_ptr + j >= len(unlabeled_indices):
                    break
                rows.append({
                    "iter_ts": ts,
                    "global_idx": int(unlabeled_indices[pool_ptr + j]),
                    "depth_entropy": float(H_img[j].item()),
                    "gt_car": int(cars[j]),
                    "gt_ped": int(peds[j]),
                    "gt_cyc": int(cycs[j]),
                })
            pool_ptr += B

        _append_rows_csv(csv_path, rows)
        if len(all_H) != len(unlabeled_indices):
            raise RuntimeError(
                f"depth-entropy count {len(all_H)} != pool size {len(unlabeled_indices)}; "
                "check dataloader order/subset."
            )
        return all_H


class BinEntropySelector(_BinEntropyBase):
    """按图像级 bin-entropy 从大到小选 Top-K，并记录统计。"""

    @torch.no_grad()
    def select(
        self,
        model: torch.nn.Module,
        dataloader,
        device: torch.device,
        unlabeled_indices: List[int],
        k: int,
    ) -> List[int]:
        scores = self._gather_and_log(model, dataloader, device, unlabeled_indices)
        order = np.argsort(scores)[::-1]  # 熵大优先
        topk_rel = order[: min(k, len(order))].tolist()
        return [unlabeled_indices[i] for i in topk_rel]


class BinEntropyMinSelector(_BinEntropyBase):
    """按图像级 bin-entropy 从小到大选 Top-K（对照组），并记录统计。"""

    @torch.no_grad()
    def select(
        self,
        model: torch.nn.Module,
        dataloader,
        device: torch.device,
        unlabeled_indices: List[int],
        k: int,
    ) -> List[int]:
        scores = self._gather_and_log(model, dataloader, device, unlabeled_indices)
        order = np.argsort(scores)        # 熵小优先
        topk_rel = order[: min(k, len(order))].tolist()
        return [unlabeled_indices[i] for i in topk_rel]


# （可选）自动注册到方法表
try:
    from active_learning.methods import _METHOD_REGISTRY
    _METHOD_REGISTRY["bin_entropy"] = BinEntropySelector
    _METHOD_REGISTRY["bin_entropy_min"] = BinEntropyMinSelector
except Exception:
    pass
