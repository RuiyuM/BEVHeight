from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
from active_learning.base import BaseActiveSelector

class _GaussBase(BaseActiveSelector):
    """Gaussian similarity over per-object features [x, y, h].

    - Target Gaussian is computed from *current labeled set* (means & cov).
    - Per-image score = average negative log-likelihood of its GT objects.
      Lower NLL => more similar to target; Higher NLL => less similar.
    - Empty images are assigned a large NLL (discourage in 'similar' mode).
    """

    def __init__(self, maximize: bool, eps: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.maximize = bool(maximize)
        self.eps = float(eps)

    # ---- data helpers ----
    def _gt_boxes_for_index(self, model, idx: int) -> np.ndarray:
        ds = model._train_dataset if model._train_dataset is not None else model._build_train_dataset()
        sample = ds[idx]

        def _to_np(a):
            if a is None:
                return None
            import numpy as np, torch
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().float().numpy()
            else:
                try:
                    a = np.asarray(a, dtype=np.float32)
                except Exception:
                    return None
            return a

        # 1) 先尝试你原先的索引（期望 sample[4] 是 gt_boxes）
        cand = None
        if isinstance(sample, (list, tuple)) and len(sample) >= 5:
            cand = _to_np(sample[4])

        def _is_boxes(arr):
            """二维、最后一维 >=6（如 CenterPoint 9 维）；且不是 (..,4,4) 矩阵"""
            if arr is None:
                return False
            import numpy as np
            if arr.ndim >= 3 and arr.shape[-2:] == (4, 4):
                return False  # 这是 4x4 矩阵，排除
            if arr.ndim > 2:
                arr = arr.reshape(-1, arr.shape[-1])
            return (arr.ndim == 2) and (arr.shape[-1] >= 6) and (arr.shape[0] > 0)

        # 2) 如果 cand 看起来不是 boxes（或是 4x4 矩阵），就扫描整个 sample 找真正的 boxes
        if not _is_boxes(cand):
            candidates = []
            if isinstance(sample, (list, tuple)):
                for it in sample:
                    arr = _to_np(it)
                    if arr is None:
                        continue
                    # 跳过 4x4 栈
                    if arr.ndim >= 3 and arr.shape[-2:] == (4, 4):
                        continue
                    # 拉平成二维
                    if arr.ndim > 2:
                        arr = arr.reshape(-1, arr.shape[-1])
                    if (arr.ndim == 2) and (arr.shape[-1] >= 6) and (arr.shape[0] > 0):
                        candidates.append(arr.astype(np.float32, copy=False))
            # 选“维度更大”的那个（优先 9维/7维格式）
            if candidates:
                candidates.sort(key=lambda a: (-a.shape[-1], -a.shape[0]))
                cand = candidates[0]
            else:
                return np.empty((0, 9), dtype=np.float32)

        # 3) 最终保证二维 (N, D)
        if cand.ndim > 2:
            cand = cand.reshape(-1, cand.shape[-1])
        # D 不足以拿 x,y,h 时返回空
        if cand.shape[-1] < 6:
            return np.empty((0, 9), dtype=np.float32)
        return cand.astype(np.float32, copy=False)

    def _features_xyh(self, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        x = boxes[:, 0:1].astype(np.float32)
        y = boxes[:, 1:2].astype(np.float32)
        h = boxes[:, 5:6].astype(np.float32)  # height length (dz)
        return np.concatenate([x, y, h], axis=1)

    def _labeled_indices(self, model) -> List[int]:
        return list(getattr(model, '_labeled_indices', []) or [])

    def _fit_target_gaussian(self, model) -> Tuple[np.ndarray, np.ndarray]:
        import numpy as np, torch
        from torch.utils.data import Subset, DataLoader

        idxs = list(getattr(model, '_labeled_indices', []) or [])
        if not idxs:
            mu = np.zeros((3,), dtype=np.float32)
            cov = np.eye(3, dtype=np.float32)
            cov = cov + self.eps * np.eye(3, dtype=np.float32)
            return mu, cov

        # 用与训练一致的数据管线：同 dataset + collate_fn + batch 组装
        ds = model._train_dataset if model._train_dataset is not None else model._build_train_dataset()
        bs = int(getattr(model, 'batch_size_per_device', 8))
        loader = DataLoader(
            Subset(ds, idxs),
            batch_size=bs,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            collate_fn=model._collate_fn_for_train(),
        )

        feats_list = []
        for batch in loader:
            *_, gt_boxes, _ = batch  # 永远用倒数两项拿 GT，避免位置变动
            for gb in gt_boxes:
                if isinstance(gb, torch.Tensor):
                    gb = gb.detach().cpu().float().numpy()
                if gb.ndim != 2 or gb.shape[1] < 6 or gb.shape[0] == 0:
                    continue
                # 取 [x, y, h] = [0, 1, 5]
                f = np.concatenate([gb[:, 0:1], gb[:, 1:2], gb[:, 5:6]], axis=1)
                feats_list.append(f)

        if not feats_list:
            mu = np.zeros((3,), dtype=np.float32)
            cov = np.eye(3, dtype=np.float32)
            cov = cov + self.eps * np.eye(3, dtype=np.float32)
            return mu, cov

        X = np.concatenate(feats_list, axis=0).astype(np.float32, copy=False)

        # 小样本/数值稳健兜底
        if X.shape[0] < 2:
            mu = X.mean(axis=0) if X.shape[0] > 0 else np.zeros((3,), np.float32)
            cov = np.eye(3, dtype=np.float32)
            cov = cov + self.eps * np.eye(3, dtype=np.float32)
            return mu, cov

        mu = X.mean(axis=0).astype(np.float32)
        cov = np.cov(X, rowvar=False)
        cov = np.asarray(cov, dtype=np.float32)

        # 再兜底一次（NaN/Inf/奇异）
        if not np.isfinite(cov).all():
            cov = np.cov(X, rowvar=False, ddof=0).astype(np.float32)
        if not np.isfinite(cov).all():
            cov = np.eye(3, dtype=np.float32)

        cov = cov + self.eps * np.eye(3, dtype=np.float32)
        return mu, cov

    def _avg_nll(self, feats: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray, logdet: float) -> float:
        if feats.size == 0:
            return 1e6  # large NLL for empty
        D = feats.shape[1]
        diff = feats - mu[None, :]
        # Mahalanobis term for each row
        m = np.einsum('ni,ij,nj->n', diff, cov_inv, diff)
        return float(0.5 * (m.mean() + logdet + D * np.log(2 * np.pi)))

    @torch.no_grad()
    def select(self, model, dataloader, device, unlabeled_indices, k):
        import numpy as np, torch
        # 先用当前 labeled 集拟合目标高斯
        mu, cov = self._fit_target_gaussian(model)
        cov = cov + self.eps * np.eye(3, dtype=np.float32)
        cov_inv = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        logdet = float(logdet) if sign > 0 else 0.0

        scores = []
        for batch in dataloader:
            _, _, _, _, _gt_boxes, _ = batch
            for gb in _gt_boxes:
                if isinstance(gb, torch.Tensor):
                    gb = gb.detach().cpu().float().numpy()
                if gb.ndim != 2 or gb.shape[1] < 6:
                    scores.append(1e6);
                    continue
                # [x,y,h]
                f = np.concatenate([gb[:, 0:1], gb[:, 1:2], gb[:, 5:6]], axis=1) if gb.shape[0] > 0 else np.empty(
                    (0, 3), np.float32)
                scores.append(self._avg_nll(f, mu, cov_inv, logdet))

        assert len(scores) == len(unlabeled_indices)
        order = np.argsort(scores)  # 低 NLL 更相似
        if self.maximize: order = order[::-1]
        topk_rel = order[: min(k, len(order))].tolist()
        return [unlabeled_indices[i] for i in topk_rel]


class GaussSimSelector(_GaussBase):
    def __init__(self, **kwargs):
        super().__init__(maximize=False, **kwargs)  # pick most similar (low NLL)

class GaussAntiSimSelector(_GaussBase):
    def __init__(self, **kwargs):
        super().__init__(maximize=True, **kwargs)   # pick least similar (high NLL)
