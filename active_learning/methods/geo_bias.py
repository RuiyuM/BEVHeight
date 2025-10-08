from __future__ import annotations
from typing import List
import numpy as np
import torch
from active_learning.base import BaseActiveSelector

class _GeoBase(BaseActiveSelector):
    """Helpers to read GT boxes and compute BEV stats.

    Assumes gt_boxes format ~ (x, y, z, dx, dy, dz, yaw, ...)
    """
    def _pc_range(self, model) -> List[float]:
        try:
            return model.head_conf['train_cfg']['point_cloud_range']
        except Exception:
            return [0, -51.2, -5, 102.4, 51.2, 3]

    def _bev_area(self, model) -> float:
        r = self._pc_range(model)
        return float((r[3] - r[0]) * (r[4] - r[1]))

    def _gt_boxes_for_index(self, model, idx: int):
        ds = model._train_dataset if model._train_dataset is not None else model._build_train_dataset()
        sample = ds[idx]
        # Expected tuple: (sweep_imgs, mats, _, _, gt_boxes, gt_labels)
        if isinstance(sample, (list, tuple)) and len(sample) >= 6:
            gt_boxes = sample[4]
        else:
            gt_boxes = None
        if isinstance(gt_boxes, torch.Tensor):
            return gt_boxes.detach().cpu().float().numpy()
        if hasattr(gt_boxes, '__len__'):
            try:
                return np.asarray(gt_boxes, dtype=np.float32)
            except Exception:
                return np.empty((0, 9), dtype=np.float32)
        return np.empty((0, 9), dtype=np.float32)

    def _centers_dims(self, boxes: np.ndarray):
        if boxes.size == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        # (x, y) and (dx, dy)
        centers = boxes[:, 0:2].astype(np.float32)
        dims = boxes[:, 3:5].astype(np.float32)
        return centers, dims

    def _max_range_xy(self, model) -> float:
        r = self._pc_range(model)
        return float(max(abs(r[0]), abs(r[1]), abs(r[3]), abs(r[4])) or 60.0)


class FarBiasSelector(_GeoBase):
    """Score = mean normalized center distance; pick DESC (farther → higher score)."""

    @torch.no_grad()
    def select(self, model, dataloader, device, unlabeled_indices, k):
        import numpy as np, torch
        maxr = self._max_range_xy(model)
        scores = []

        for batch in dataloader:
            _, _, _, _, _gt_boxes, _ = batch
            for gb in _gt_boxes:
                if isinstance(gb, torch.Tensor):
                    gb = gb.detach().cpu().float().numpy()
                if gb.ndim != 2 or gb.shape[1] < 6 or gb.shape[0] == 0:
                    scores.append(0.0);
                    continue
                centers = gb[:, :2]
                d = np.sqrt(centers[:, 0] ** 2 + centers[:, 1] ** 2)
                s = float(np.clip(d / maxr, 0.0, 1.0).mean())
                scores.append(s)

        assert len(scores) == len(unlabeled_indices)
        order = np.argsort(scores)[::-1]  # 远的优先
        topk_rel = order[: min(k, len(order))].tolist()
        return [unlabeled_indices[i] for i in topk_rel]


class NearOccupancySelector(_GeoBase):
    """Score = sum BEV footprint area of *near* objects / total BEV area; pick DESC.

    Near threshold can be passed via kwargs['near_thresh'] (meters), default 30.0.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.near_thresh = float(self.kwargs.get('near_thresh', 30.0))

    @torch.no_grad()
    def select(self, model, dataloader, device, unlabeled_indices, k):
        import numpy as np, torch
        bev_area = max(1e-6, self._bev_area(model))
        T = float(self.near_thresh)
        scores = []

        for batch in dataloader:
            _, _, _, _, _gt_boxes, _ = batch
            for gb in _gt_boxes:
                if isinstance(gb, torch.Tensor):
                    gb = gb.detach().cpu().float().numpy()
                if gb.ndim != 2 or gb.shape[1] < 6 or gb.shape[0] == 0:
                    scores.append(0.0);
                    continue
                centers = gb[:, :2]
                dims = gb[:, 3:5]  # dx, dy
                d = np.sqrt(centers[:, 0] ** 2 + centers[:, 1] ** 2)
                m = d <= T
                if not np.any(m):
                    scores.append(0.0);
                    continue
                occ = float(np.sum(np.clip(dims[m, 0], 0, None) * np.clip(dims[m, 1], 0, None)))
                scores.append(float(np.clip(occ / bev_area, 0.0, 1.0)))

        assert len(scores) == len(unlabeled_indices)
        order = np.argsort(scores)[::-1]  # 占比大的优先
        topk_rel = order[: min(k, len(order))].tolist()
        return [unlabeled_indices[i] for i in topk_rel]

